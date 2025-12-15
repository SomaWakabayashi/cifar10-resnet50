import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 自作モジュールのインポート
from utils.common import fix_seed
from utils.transforms import mixup_data, cutmix_data, mix_criterion
from utils.loss import LabelSmoothingLoss

def get_transforms(augment_mode='simple'):
    """
    データ拡張の設定を切り替える関数
    simple: 正規化のみ（Week1の状態）
    standard: 一般的なデータ拡張（Crop + Flip）※これを基本とする
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # テスト用は常に正規化のみ
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if augment_mode == 'simple':
        # Week 1のベースライン（拡張なし）
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        # standard, mixup, cutmix 共通のベース拡張
        # Mixup等は画像バッチ作成後に適用するため、ここでは幾何学的変換のみ行う
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # 4px枠を足してランダムに切り出す
            transforms.RandomHorizontalFlip(),    # 50%で左右反転
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transform_train, transform_test

def get_data_loaders(batch_size, augment_mode='standard', num_workers=2):
    transform_train, transform_test = get_transforms(augment_mode)

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_set.classes

def get_model(num_classes=10):
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, augment_method='standard'):
    """augment_methodによって学習ロジックを分岐"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        # --- Data Augmentation Logic ---
        if augment_method == 'mixup':
            # Mixup適用
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0, device=device)
            outputs = model(images)
            loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            # 精度計算用（より支配的なラベルの方を正解とみなす簡易計算）
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()

        elif augment_method == 'cutmix':
            # Cutmix適用
            images, targets_a, targets_b, lam = cutmix_data(images, labels, beta=1.0, device=device)
            outputs = model(images)
            loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            
        else:
            # Standard / Simple (通常学習)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        # -------------------------------

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': running_loss / (total / loader.batch_size)})

    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), all_preds, all_labels

def save_plots(history, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'acc_curve.png'))
    plt.close()

def save_confusion_matrix(all_labels, all_preds, classes, save_dir):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def save_classification_report(all_labels, all_preds, classes, save_dir):
    report = classification_report(all_labels, all_preds, target_names=classes)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name (Required)')
    parser.add_argument('--augment', type=str, default='standard', choices=['simple', 'standard', 'mixup', 'cutmix'], help='Augmentation method')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing value (0.0 to disable)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50) # デフォルトを多めに設定
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device)
    
    save_dir = os.path.join('runs', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 設定を保存
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    print(f"Using device: {device}")
    print(f"Augmentation: {args.augment} | Label Smoothing: {args.label_smoothing}")

    # 1. Data
    train_loader, test_loader, classes = get_data_loaders(args.batch_size, augment_mode=args.augment)

    # 2. Model
    model = get_model().to(device)

    # 3. Loss & Optimizer
    if args.label_smoothing > 0.0:
        print(f"Using LabelSmoothingLoss (smoothing={args.label_smoothing})")
        criterion = LabelSmoothingLoss(classes=len(classes), smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # 学習率スケジューラ（CosineAnnealing）を追加
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"Start training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, augment_method=args.augment)
        val_loss, val_preds, val_labels = evaluate(model, test_loader, criterion, device)
        val_acc = 100. * sum(np.array(val_preds) == np.array(val_labels)) / len(val_labels)
        
        scheduler.step() # LR更新

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{args.epochs}] LR: {scheduler.get_last_lr()[0]:.4f} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # 5. Save & Report
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, 'history.csv'), index=False)
    save_plots(history, save_dir)
    save_confusion_matrix(val_labels, val_preds, classes, save_dir)
    save_classification_report(val_labels, val_preds, classes, save_dir)
    
    print(f"✅ Training Complete. Saved to {save_dir}")

if __name__ == '__main__':
    main()