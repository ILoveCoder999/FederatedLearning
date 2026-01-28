import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, Subset
from preprocessing import FederatedDataBuilder
from taskarithmetic import SparseSGDM, compute_fisher_sensitivity, calibrate_masks
from fed_avg_iid import DINOCIFAR100Fixed, fed_avg_aggregate # 复用你之前的模型和聚合函数

# ============================================================
# 1. 本地训练函数 (集成了 Task Arithmetic)
# ============================================================
def local_train_task_arithmetic(model, train_dataset, client_indices, device, 
                                 sparsity_ratio=0.1, local_epochs=4):
    """
    客户端本地训练：包含掩码校准和稀疏微调
    """
    model.train()
    model.to(device)
    
    # 准备本地数据
    local_sub = Subset(train_dataset, list(client_indices))
    local_loader = DataLoader(local_sub, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # --- 阶段 A: 掩码校准 (Mask Calibration) ---
    # 根据项目要求：识别最不敏感 (least-sensitive) 的参数
    # 使用一轮 (num_batches=len(local_loader)) 即可代表本地数据的敏感度
    sensitivity_scores = compute_fisher_sensitivity(
        model, local_loader, criterion, device, num_batches=len(local_loader)
    )
    
    masks = calibrate_masks(
        sensitivity_scores, 
        sparsity_ratio=sparsity_ratio, 
        keep_least_sensitive=True # 项目 3.3 要求
    )

    # --- 阶段 B: 稀疏微调 (Sparse Fine-tuning) ---
    # 使用你实现的 SparseSGDM
    optimizer = SparseSGDM(
        model.parameters(), 
        lr=0.01, 
        momentum=0.9, 
        masks=masks
    )

    for epoch in range(local_epochs):
        for inputs, labels in local_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # 这里会自动应用掩码

    return model.state_dict(), len(local_sub)

# ============================================================
# 2. 主联邦训练循环
# ============================================================
def run_fed_iid_task_arithmetic(rounds=50, num_clients=100, sampling_rate=0.1, sparsity=0.1):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据准备
    builder = FederatedDataBuilder()
    # 获取 IID 划分
    dict_users = builder.build_iid() 
    test_loader = DataLoader(builder.test_dataset, batch_size=128, shuffle=False)

    # 初始化全局模型
    global_model = DINOCIFAR100Fixed(num_classes=100).to(DEVICE)
    
    history = {"accuracy": [], "loss": []}

    for r in range(rounds):
        print(f"\n--- Round {r+1}/{rounds} (Sparsity: {sparsity}) ---")
        
        local_weights = []
        local_counts = []
        
        # 随机选择客户端
        m = max(int(sampling_rate * num_clients), 1)
        selected_clients = np.random.choice(range(num_clients), m, replace=False)

        for client_id in selected_clients:
            # 深拷贝全局模型到本地
            local_model = copy.deepcopy(global_model)
            
            # 本地任务算术训练
            w, count = local_train_task_arithmetic(
                local_model, 
                builder.train_dataset, 
                dict_users[client_id], 
                DEVICE,
                sparsity_ratio=sparsity
            )
            
            local_weights.append(w)
            local_counts.append(count)

        # 聚合 (FedAvg)
        global_weights = fed_avg_aggregate(global_model, local_weights, local_counts)
        global_model.load_state_dict(global_weights)

        # 全局评估
        acc, loss = evaluate(global_model, test_loader, DEVICE)
        history["accuracy"].append(acc)
        history["loss"].append(loss)
        print(f"Round {r+1} Global Test Acc: {acc:.2f}%")

    return history

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total, total_loss / len(loader)

if __name__ == "__main__":
    import numpy as np
    # 运行实验
    run_fed_iid_task_arithmetic(rounds=20, sparsity=0.1)
