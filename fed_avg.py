import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# 假设之前的代码保存在以下模块中 (根据你实际的文件名调整)
from preprocessing import FederatedDataBuilder
from centralized_model import DINOCIFAR100

def fed_avg_aggregate(global_model, local_weights, client_sample_counts):
    """
    Performs the weighted averaging of local model weights.
    w_global = sum(n_k * w_k) / sum(n_k)
    """
    # Create a deep copy of the global model state to update
    global_dict = global_model.state_dict()

    # Calculate total samples in this round for weighted average
    total_samples = sum(client_sample_counts)

    # Initialize the aggregated dictionary
    # We take the first local model as the base (scaled by its weight)
    first_weights = local_weights[0]
    first_ratio = client_sample_counts[0] / total_samples

    for k in global_dict.keys():
        # Handle strict type checking for scalars (long/float)
        if 'num_batches_tracked' in k:
            global_dict[k] = first_weights[k]
        else:
            global_dict[k] = first_weights[k] * first_ratio

    # Add the rest of the models
    for i in range(1, len(local_weights)):
        ratio = client_sample_counts[i] / total_samples
        weights = local_weights[i]
        for k in global_dict.keys():
            if 'num_batches_tracked' not in k:
                global_dict[k] += weights[k] * ratio

    return global_dict

class LocalClient:
    """
    Simulates a local client training process.
    """
    def __init__(self, client_id, dataset, indices, device, model_class):
        self.client_id = client_id
        self.dataset = dataset
        self.indices = indices
        self.device = device
        self.model_class = model_class

        # Create a local dataloader
        # Since J (steps) is small (4), batch size matters.
        # Standard FL batch size is often 10-50. Let's pick 32 or 64.
        self.trainloader = DataLoader(
            Subset(dataset, list(indices)),
            batch_size=32,
            shuffle=True
        )

    def train(self, global_weights, local_steps=4, lr=0.01):
        """
        Runs local training for J steps (not epochs).
        """
        # 1. Initialize local model with global weights
        local_model = self.model_class(num_classes=100).to(self.device)
        local_model.load_state_dict(global_weights)
        local_model.train()

        # 2. Setup Optimizer (SGD is standard for FedAvg)
        # Note: In standard FedAvg, optimizer state is NOT passed between rounds.
        optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # 3. Local Training Loop (J steps) [cite: 59]
        step_count = 0
        epoch_loss = []

        # Create an iterator that resets if we run out of data
        iterator = iter(self.trainloader)

        while step_count < local_steps:
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                # Restart iterator if dataset is exhausted (though unlikely for J=4)
                iterator = iter(self.trainloader)
                inputs, targets = next(iterator)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            step_count += 1

        return local_model.state_dict(), sum(epoch_loss)/len(epoch_loss)

def evaluate_global(model, test_loader, device):
    """Evaluate global model on server test set"""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return loss_sum / len(test_loader), 100. * correct / total

def run_fedavg_experiment():
    # ---------------------------------------------------------
    # Configuration [cite: 59]
    # ---------------------------------------------------------
    K = 100             # Total clients
    C = 0.1             # Fraction of clients
    J = 4               # Local steps (NOT Epochs)
    ROUNDS = 50         # "Proper number of rounds" [cite: 60] - Adjust based on convergence
    LR = 0.01           # Learning rate
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running FedAvg | K={K}, C={C}, J={J} steps | Device: {DEVICE}")

    # 1. Data Preparation (Task 3.1)
    # Using IID sharding as requested for the first baseline [cite: 59]
    # Note: Ensure FederatedDataBuilder is imported/available
    data_builder = FederatedDataBuilder(val_split_ratio=0.1, K=K)
    client_dict = data_builder.get_iid_partition() # Get IID indices

    test_loader = DataLoader(data_builder.test_dataset, batch_size=64, shuffle=False)

    # 2. Global Model Initialization
    global_model = DINOCIFAR100(num_classes=100).to(DEVICE)

    # History for plotting
    history = {'loss': [], 'accuracy': []}

    # 3. Federated Training Loop
    for r in range(ROUNDS):
        print(f"\n--- Round {r+1}/{ROUNDS} ---")

        # a. Client Selection
        # m = max(C * K, 1) -> Select 10 clients
        m = max(int(C * K), 1)
        selected_clients = np.random.choice(range(K), m, replace=False)
        print(f"Selected clients: {selected_clients}")

        local_weights = []
        client_sample_counts = []

        # b. Local Training (Sequential Simulation)
        # Copy global weights to CPU to avoid reference issues during iteration
        global_weights = copy.deepcopy(global_model.state_dict())

        for client_idx in selected_clients:
            # Create client interface
            client = LocalClient(
                client_id=client_idx,
                dataset=data_builder.train_dataset,
                indices=client_dict[client_idx],
                device=DEVICE,
                model_class=DINOCIFAR100
            )

            # Train locally
            w_local, loss_local = client.train(global_weights, local_steps=J, lr=LR)

            local_weights.append(w_local)
            client_sample_counts.append(len(client_dict[client_idx]))

            # Optional: Print local progress (comment out for speed)
            # print(f"  Client {client_idx} loss: {loss_local:.4f}")

        # c. Aggregation (FedAvg) [cite: 4]
        # Update global model with weighted average of local weights
        new_weights = fed_avg_aggregate(global_model, local_weights, client_sample_counts)
        global_model.load_state_dict(new_weights)

        # d. Evaluation
        # Evaluate global model on the held-out test set
        test_loss, test_acc = evaluate_global(global_model, test_loader, DEVICE)
        history['loss'].append(test_loss)
        history['accuracy'].append(test_acc)

        print(f"Global Model Stats -> Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")

    # 4. Plot Results
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, ROUNDS+1), history['accuracy'], marker='o')
    plt.title(f'FedAvg (IID) Performance (J={J})')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.show()
