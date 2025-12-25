import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# 前回のコードからインポート
from train_cifar10_resnet50_advanced import (
    get_transforms, get_optimized_resnet50, train_one_epoch, evaluate, fix_seed
)
from utils.loss import LabelSmoothingLoss

def smart_resume_training():
    exp_name = 'resnet50_cifar10_all_aug_reg'
    save_dir = os.path.join('runs', exp_name)
    checkpoint_path = os.path.join(save_dir, 'model.pth')
    history_path = os.path.join(save_dir, 'history.csv')
    
    TOTAL_GOAL_EPOCHS = 100
    FIXED_BASE_LR = 0.02 # resume開始時のLR

    fix_seed(42)
    device = torch.device('cuda')

    # 1. 履歴の読み込みと現在地の特定
    if not os.path.exists(history_path):
        print("History file not found. Cannot resume smart.")
        return
    
    history_df = pd.read_csv(history_path)
    current_epoch = len(history_df)
    remaining_epochs = TOTAL_GOAL_EPOCHS - current_epoch

    if remaining_epochs <= 0:
        print(f"Already reached goal of {TOTAL_GOAL_EPOCHS} epochs.")
        return

    print(f">>> Current progress: {current_epoch} epochs. Goal: {TOTAL_GOAL_EPOCHS}.")
    print(f">>> Remaining: {remaining_epochs} epochs.")

    # 2. データとモデルの準備
    import torchvision
    from torch.utils.data import DataLoader
    train_transform, test_transform = get_transforms(mode='advanced')
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = get_optimized_resnet50(num_classes=10).to(device)
    
    # 重みのロード（コンパイル済みのプレフィックス対応）
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    # 3. オプティマイザとスケジューラの調整
    optimizer = optim.SGD(model.parameters(), lr=FIXED_BASE_LR, momentum=0.9, weight_decay=1e-3)
    criterion = LabelSmoothingLoss(classes=10, smoothing=0.1)
    
    # 40〜100エポック（計60歩）のスケジューラを作成し、現在地まで進める
    full_resume_period = 60 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=full_resume_period)
    
    # スケジューラを現在地までスキップ（例：63エポックなら、40から数えて23歩進める）
    passed_in_resume = current_epoch - 40
    for _ in range(passed_in_resume):
        scheduler.step()

    scaler = torch.amp.GradScaler('cuda')
    history = {col: history_df[col].tolist() for col in history_df.columns}

    # 4. 学習ループ
    for _ in range(remaining_epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_mixup=True)
        v_loss, v_preds, v_labels = evaluate(model, test_loader, criterion, device)
        v_acc = 100. * sum(np.array(v_preds) == np.array(v_labels)) / len(v_labels)
        scheduler.step()

        for k, v in zip(['train_loss', 'train_acc', 'val_loss', 'val_acc'], [t_loss, t_acc, v_loss, v_acc]):
            history[k].append(v)
        
        actual_epoch = len(history['val_acc'])
        print(f"Epoch [{actual_epoch}/100] Val Acc: {v_acc:.2f}% (LR: {scheduler.get_last_lr()[0]:.6f})")

        # 保存
        if actual_epoch % 5 == 0 or actual_epoch == 100:
            torch.save(model.state_dict(), checkpoint_path)
            pd.DataFrame(history).to_csv(history_path, index=False)

    print(f"✅ Successfully reached 100 epochs.")

if __name__ == '__main__':
    smart_resume_training()