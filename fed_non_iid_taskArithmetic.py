import torch
import torch.nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 导入你之前的模块
from preprocessing import FederatedDataBuilder
from taskarithmetic import SparseSGDM, compute_fisher_sensitivity, calibrate_masks
from fed_avg_non_iid import (
    DINOCIFAR100, 
    create_non_iid_partition, 
    fed_avg_aggregate, 
    evaluate_global,
    LocalClient # 我们将对其进行继承或修改
)

# ============================================================
# 1. 强化版本地客户端 (集成 Task Arithmetic)
# ============================================================
class TaskArithmeticClient(LocalClient):
    """
    支持任务算术的本地客户端：
    1. 动态校准掩码 (Calibration)
    2. 执行稀疏梯度更新 (Sparse Fine-tuning)
    """
    def train_with_mask(self, global_weights, local_steps, lr=0.01, sparsity_ratio=0.1):
        # 初始化模型并加载全局权重
        local_model = DINOCIFAR100(num_classes=100).to(self.device)
        local_model.load_state_dict(global_weights, strict=False)
        
        criterion = nn.CrossEntropyLoss()

        # --- 步骤 1: 校准梯度掩码 (项目要求 3.3) ---
        # 在本地数据上运行若干 batch 来计算 Fisher 信息敏感度
        # 我们使用整个本地数据集的一轮迭代来校准
        calib_batches = max(1, len(self.trainloader))
        sensitivity_scores = compute_fisher_sensitivity(
            local_model, self.trainloader, criterion, self.device, num_batches=calib_batches
        )
        
        # 识别最不敏感 (least-sensitive) 的参数进行更新
        masks = calibrate_masks(
            sensitivity_scores, 
            sparsity_ratio=sparsity_ratio, 
            keep_least_sensitive=True
        )

        # --- 步骤 2: 使用 SparseSGDM 进行稀疏微调 ---
        optimizer = SparseSGDM(
            local_model.head.parameters(), # 仅训练 Head [cite: 67]
            lr=lr,
            momentum=0.9,
            masks=masks
        )

        local_model.train()
        step_count = 0
        losses = []
        iterator = iter(self.trainloader)

        while step_count < local_steps:
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(self.trainloader)
                inputs, targets = next(iterator)

            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 这里的 optimizer.step() 会自动应用刚才生成的掩码
            optimizer.step()

            losses.append(loss.item())
            step_count += 1

        return local_model.state_dict(), sum(losses)/len(losses)

# ============================================================
# 2. 实验运行主函数
# ============================================================
def run_fed_non_iid_task_arithmetic(Nc=1, sparsity=0.1, rounds=50):
    """
    在 Non-IID 环境下运行带任务算术的联邦学习
    Nc: 每个客户端拥有的类别数 (Nc=1 是最极端的 Non-IID) [cite: 44, 45]
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K, C, J = 100, 0.1, 8 # 使用项目定义的参数 [cite: 59, 62]
    
    # 数据与分区
    builder = FederatedDataBuilder(K=K)
    client_dict = create_non_iid_partition(builder.train_dataset, K, Nc)
    test_loader = DataLoader(builder.test_dataset, batch_size=128, shuffle=False)

    # 全局模型
    global_model = DINOCIFAR100(num_classes=100).to(DEVICE)
    history = {'accuracy': [], 'round': []}

    m = max(int(C * K), 1)
    
    for r in range(rounds):
        selected_clients = np.random.choice(range(K), m, replace=False)
        local_weights, local_counts = [], []
        global_w_copy = copy.deepcopy(global_model.state_dict())

        for client_idx in selected_clients:
            client = TaskArithmeticClient(
                client_id=client_idx,
                dataset=builder.train_dataset,
                indices=client_dict[client_idx],
                device=DEVICE
            )
            
            # 执行任务算术训练
            w_local, _ = client.train_with_mask(
                global_w_copy, local_steps=J, sparsity_ratio=sparsity
            )
            
            local_weights.append(w_local)
            local_counts.append(len(client_dict[client_idx]))

        # 聚合更新
        new_global_w = fed_avg_aggregate(global_model, local_weights, local_counts)
        global_model.load_state_dict(new_global_w)

        # 测试
        _, acc = evaluate_global(global_model, test_loader, DEVICE)
        history['accuracy'].append(acc)
        history['round'].append(r + 1)
        
        print(f"Round {r+1}/{rounds} (Nc={Nc}, Sparsity={sparsity}) -> Test Acc: {acc:.2f}%")

    return history

if __name__ == "__main__":
    # 示例运行：极端的 Non-IID 环境
    run_fed_non_iid_task_arithmetic(Nc=1, sparsity=0.1, rounds=30)
