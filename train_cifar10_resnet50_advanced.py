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
from sklearn.metrics import confusion_matrix

from utils.common import fix_seed
from utils.transforms import mixup_data, mix_criterion
from utils.loss import LabelSmoothingLoss

def get_transforms(mode='advanced'):
    """
    高度なデータ拡張セットを定義
    - RandAugment: 自動画像変換
    - ColorJitter: 色彩の頑健性
    - RandomErasing: 物体欠損への耐性
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if mode == 'advanced':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return train_transform, test_transform

def get_optimized_resnet50(num_classes=10):
    """
    CIFAR-10 (32x32) 向けにアーキテクチャを最適化したResNet50
    - 最初の7x7 Convを3x3に変更し情報の消失を防止
    - MaxPoolを削除し低解像度を維持
    """
    model = torchvision.models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_mixup=True):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda'):
            if use_mixup:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0, device=device)
                outputs = model(images)
                loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader), all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description='Advanced ResNet50 Training for CIFAR-10')
    parser.add_argument('--exp_name', type=str, default='resnet50_cifar10_all_aug_reg')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--use_mixup', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=1e-3)
    args = parser.parse_args()

    fix_seed(42)
    device = torch.device('cuda')
    save_dir = os.path.join('runs', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Data
    train_transform, test_transform = get_transforms(mode='advanced')
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    classes = train_set.classes

    # 2. Model, Criterion, Optimizer
    model = get_optimized_resnet50(num_classes=10).to(device)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    criterion = LabelSmoothingLoss(classes=len(classes), smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    # 3. Training Loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(args.epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, args.use_mixup)
        v_loss, v_preds, v_labels = evaluate(model, test_loader, criterion, device)
        v_acc = 100. * sum(np.array(v_preds) == np.array(v_labels)) / len(v_labels)
        scheduler.step()
        
        for k, v in zip(history.keys(), [t_loss, t_acc, v_loss, v_acc]):
            history[k].append(v)
        print(f"Epoch [{epoch+1}/{args.epochs}] Val Acc: {v_acc:.2f}%")

    # 4. Save Artifacts
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    pd.DataFrame(history).to_csv(os.path.join(save_dir, 'history.csv'), index=False)
    
    cm = confusion_matrix(v_labels, v_preds)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png')); plt.close()
    
    print(f"✅ Experiment '{args.exp_name}' Complete.")

if __name__ == '__main__':
    main()