import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime

# ============================================================
# 1. Model Definition (with Dropout)
# ============================================================
class DINOCIFAR100(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.1):
        """
        DINO ViT-S/16 model for CIFAR-100 classification.

        Args:
            num_classes: Number of output classes (100 for CIFAR-100)
            dropout_rate: Dropout probability for regularization
        """
        super(DINOCIFAR100, self).__init__()
        print("Downloading/Loading DINO ViT-S/16...")
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.embed_dim = 384

        # Add Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        """Forward pass through the network"""
        features = self.backbone(x)
        features = self.dropout(features)
        return self.head(features)


# ============================================================
# 2. Training & Evaluation Functions
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: Neural network model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/cpu)

    Returns:
        avg_loss: Average loss for this epoch
        accuracy: Training accuracy (%)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation/test set.

    Args:
        model: Neural network model
        loader: Validation/test data loader
        criterion: Loss function
        device: Device (cuda/cpu)

    Returns:
        avg_loss: Average loss
        accuracy: Accuracy (%)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ============================================================
# 3. Warmup + Cosine Decay Learning Rate Scheduler
# ============================================================
def get_lr_schedule(optimizer, warmup_epochs=5, total_epochs=50):
    """
    Create a learning rate scheduler with Warmup + Cosine Decay.

    - First warmup_epochs: Linear warmup from 0 to initial LR
    - Remaining epochs: Cosine annealing decay

    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup phase: Linear growth from 0 to 1
            return epoch / warmup_epochs
        else:
            # Cosine Decay phase: Smooth decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 4. Checkpointing Functions
# ============================================================
def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, history,
                   checkpoint_dir='checkpoints', filename='checkpoint.pth'):
    """
    Save training checkpoint.

    Args:
        epoch: Current epoch number
        model: Model state
        optimizer: Optimizer state
        scheduler: Scheduler state
        best_acc: Best validation accuracy so far
        history: Training history dictionary
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'history': history
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into

    Returns:
        start_epoch: Epoch to resume from
        best_acc: Best accuracy from checkpoint
        history: Training history
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, 0.0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'] + 1, checkpoint['best_acc'], checkpoint['history']


# ============================================================
# 5. Experiment Logging
# ============================================================
class ExperimentLogger:
    """Simple logger for tracking experiments"""

    def __init__(self, log_dir='logs', experiment_name=None):
        """
        Initialize experiment logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (auto-generated if None)
        """
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f'centralized_{timestamp}'

        self.log_file = os.path.join(log_dir, f'{experiment_name}.json')
        self.metrics = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'hyperparameters': {},
            'epochs': []
        }

    def log_hyperparameters(self, hparams):
        """Log hyperparameters"""
        self.metrics['hyperparameters'] = hparams

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """Log metrics for one epoch"""
        self.metrics['epochs'].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': lr
        })

    def log_final_results(self, test_loss, test_acc, total_epochs):
        """Log final test results"""
        self.metrics['end_time'] = datetime.now().isoformat()
        self.metrics['total_epochs'] = total_epochs
        self.metrics['test_loss'] = test_loss
        self.metrics['test_acc'] = test_acc

    def save(self):
        """Save logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Experiment log saved to {self.log_file}")


# ============================================================
# 6. Main Training Function (Enhanced)
# ============================================================
def run_centralized_baseline(config=None, seed=42, resume_from=None):
    """
    Main training function with all optimizations and best practices.

    Args:
        config: Dictionary with hyperparameters (if None, uses defaults)
        seed: Random seed for reproducibility
        resume_from: Path to checkpoint to resume from

    Returns:
        test_acc: Final test accuracy
        history: Training history
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # ========== Hyperparameters ==========
    if config is None:
        config = {
            'batch_size': 128,
            'epochs': 50,
            'lr': 0.0001,
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'dropout_rate': 0.1,
            'warmup_epochs': 5,
            'patience': 10
        }

    BATCH_SIZE = config['batch_size']
    EPOCHS = config['epochs']
    LR = config['lr']
    MOMENTUM = config['momentum']
    WEIGHT_DECAY = config['weight_decay']
    DROPOUT_RATE = config['dropout_rate']
    WARMUP_EPOCHS = config['warmup_epochs']
    PATIENCE = config['patience']

    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Random seed: {seed}")

    # ========== Initialize Logger ==========
    logger = ExperimentLogger(experiment_name=f'centralized_seed{seed}')
    logger.log_hyperparameters(config)

    # ========== Data Preparation ==========
    import torchvision.transforms as transforms

    # Training set: Apply data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761))
    ])

    # Validation/Test set: No augmentation
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                           (0.2675, 0.2565, 0.2761))
    ])

    # Load datasets
    from preprocessing import FederatedDataBuilder
    data_builder = FederatedDataBuilder(val_split_ratio=0.1)

    # Apply augmentation to training set
    data_builder.train_dataset.dataset.transform = transform_train

    train_loader = torch.utils.data.DataLoader(
        data_builder.train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        data_builder.val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        data_builder.test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"\nDataset sizes:")
    print(f"  Training: {len(data_builder.train_dataset)}")
    print(f"  Validation: {len(data_builder.val_dataset)}")
    print(f"  Test: {len(data_builder.test_dataset)}")

    # ========== Model, Optimizer, Loss Function ==========
    model = DINOCIFAR100(num_classes=100, dropout_rate=DROPOUT_RATE).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                         weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_lr_schedule(optimizer, warmup_epochs=WARMUP_EPOCHS,
                               total_epochs=EPOCHS)

    # ========== Resume from checkpoint if specified ==========
    start_epoch = 0
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if resume_from is not None:
        start_epoch, best_val_acc, history = load_checkpoint(
            resume_from, model, optimizer, scheduler
        )
        print(f"Resumed from epoch {start_epoch} with best acc {best_val_acc:.2f}%")

    # ========== Early Stopping Setup ==========
    no_improve_count = 0
    checkpoint_dir = 'checkpoints'
    best_model_path = os.path.join(checkpoint_dir, f'best_model_seed{seed}.pth')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========== Training Loop ==========
    print(f"\n{'='*60}")
    print(f"Starting centralized training for {EPOCHS} epochs...")
    print(f"Hyperparameters: LR={LR}, Batch={BATCH_SIZE}, WD={WEIGHT_DECAY}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, EPOCHS):
        # Train for one epoch
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion,
                                       optimizer, DEVICE)

        # Validate
        v_loss, v_acc = evaluate(model, val_loader, criterion, DEVICE)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update learning rate
        scheduler.step()

        # Record history
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        # Log to experiment logger
        logger.log_epoch(epoch + 1, t_loss, t_acc, v_loss, v_acc, current_lr)

        # Early Stopping check
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_model_path)
            no_improve_count = 0
            print(f"✓ Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | "
                  f"Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | "
                  f"Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}% | "
                  f"NEW BEST! ⭐")
        else:
            no_improve_count += 1
            print(f"  Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | "
                  f"Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | "
                  f"Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}% | "
                  f"No improvement: {no_improve_count}/{PATIENCE}")

        # Save periodic checkpoint (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc,
                          history, checkpoint_dir,
                          f'checkpoint_epoch{epoch+1}_seed{seed}.pth')

        # Stop training if no improvement for PATIENCE epochs
        if no_improve_count >= PATIENCE:
            print(f"\n⚠ Early stopping at epoch {epoch+1}")
            break

    # ========== Load Best Model for Final Test ==========
    print(f"\n{'='*60}")
    print("Training completed! Loading best model for testing...")
    print(f"{'='*60}\n")

    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

    print(f" Final Results:")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")

    # Log final results
    logger.log_final_results(test_loss, test_acc, epoch + 1)
    logger.save()

    # ========== Plot Training Curves ==========
    plot_results(history, save_path=f'figures/training_curves_seed{seed}.png')

    return test_acc, history


# ============================================================
# 7. Multiple Runs for Statistical Significance
# ============================================================
def run_multiple_experiments(num_runs=3, config=None):
    """
    Run multiple experiments with different random seeds.

    Args:
        num_runs: Number of independent runs
        config: Hyperparameter configuration

    Returns:
        results: List of test accuracies from each run
    """
    print(f"\n{'='*60}")
    print(f"Running {num_runs} independent experiments")
    print(f"{'='*60}\n")

    results = []

    for run in range(num_runs):
        seed = 42 + run
        print(f"\n{'#'*60}")
        print(f"# RUN {run+1}/{num_runs} (seed={seed})")
        print(f"{'#'*60}\n")

        test_acc, _ = run_centralized_baseline(config=config, seed=seed)
        results.append(test_acc)

    # Calculate statistics
    mean_acc = np.mean(results)
    std_acc = np.std(results)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (across {num_runs} runs)")
    print(f"{'='*60}")
    print(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Individual runs: {[f'{acc:.2f}%' for acc in results]}")
    print(f"{'='*60}\n")

    # Save aggregated results
    os.makedirs('logs', exist_ok=True)
    with open('logs/multiple_runs_summary.json', 'w') as f:
        json.dump({
            'num_runs': num_runs,
            'results': results,
            'mean': mean_acc,
            'std': std_acc,
            'config': config
        }, f, indent=2)

    return results


# ============================================================
# 8. Hyperparameter Search
# ============================================================
def hyperparameter_search():
    """
    Perform hyperparameter search over common parameters.
    This is a simple grid search - you can expand it as needed.

    Returns:
        best_config: Best hyperparameter configuration
        all_results: Results from all configurations
    """
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH")
    print(f"{'='*60}\n")

    # Define search space (simplified for demonstration)
    search_space = {
        'lr': [0.0001, 0.0005, 0.001],
        'batch_size': [64, 128],
        'weight_decay': [1e-4, 5e-4]
    }

    # Base configuration
    base_config = {
        'epochs': 30,  # Reduced for search
        'momentum': 0.9,
        'dropout_rate': 0.1,
        'warmup_epochs': 5,
        'patience': 10
    }

    all_results = []
    best_acc = 0
    best_config = None

    # Grid search
    for lr in search_space['lr']:
        for batch_size in search_space['batch_size']:
            for weight_decay in search_space['weight_decay']:
                config = base_config.copy()
                config.update({
                    'lr': lr,
                    'batch_size': batch_size,
                    'weight_decay': weight_decay
                })

                print(f"\nTesting config: LR={lr}, BS={batch_size}, WD={weight_decay}")
                test_acc, _ = run_centralized_baseline(config=config, seed=42)

                all_results.append({
                    'config': config,
                    'test_acc': test_acc
                })

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_config = config
                    print(f"✓ New best config! Acc={test_acc:.2f}%")

    # Save search results
    with open('logs/hyperparameter_search.json', 'w') as f:
        json.dump({
            'best_config': best_config,
            'best_acc': best_acc,
            'all_results': all_results
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"{'='*60}\n")

    return best_config, all_results


# ============================================================
# 9. Plotting Function (Enhanced)
# ============================================================
def plot_results(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dictionary containing training history
        save_path: Optional path to save figure
    """
    os.makedirs('figures', exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Add best validation accuracy marker
    best_val_idx = np.argmax(history['val_acc'])
    best_val_acc = history['val_acc'][best_val_idx]
    axes[1].axvline(x=best_val_idx+1, color='green', linestyle='--',
                   alpha=0.5, label=f'Best: {best_val_acc:.2f}%')
    axes[1].legend(fontsize=11)

    plt.tight_layout()

    if save_path is None:
        save_path = 'figures/training_curves.png'

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # Option 1: Single run with default config
    print("Option 1: Single run with default configuration")
    test_acc, history = run_centralized_baseline()

    # Option 2: Multiple runs for statistical significance
    # print("Option 2: Multiple runs for statistical significance")
    results = run_multiple_experiments(num_runs=3)

    # Option 3: Hyperparameter search (commented out - takes time)
    # print("Option 3: Hyperparameter search")
    best_config, all_results = hyperparameter_search()
    # print(f"Best config: {best_config}")
