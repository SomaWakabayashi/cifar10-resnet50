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

def get_transforms(augment_mode='standard'):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if augment_mode == 'simple':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif augment_mode == 'randaug':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        # standard
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transform_train, transform_test

def get_data_loaders(batch_size, augment_mode='standard', num_workers=2):
    transform_train, transform_test = get_transforms(augment_mode)
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, train_set.classes

def get_model(num_classes=10):
    # 標準のResNet50をロード
    model = torchvision.models.resnet50(weights=None)
    
    # --- CIFAR-10 向けの最適化 ---
    # 1. 最初の畳み込み層を 7x7 stride=2 から 3x3 stride=1 に変更
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 2. 最初の MaxPool (画像をさらに1/2にする) を無効化
    model.maxpool = nn.Identity()
    
    # 最終層の付け替え
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, augment_method='standard'):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            if augment_method == 'mixup':
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=1.0, device=device)
                outputs = model(images)
                loss = mix_criterion(criterion, outputs, targets_a, targets_b, lam)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            elif augment_method == 'cutmix':
                images, targets_a, targets_b, lam = cutmix_data(images, labels, beta=1.0, device=device)
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
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader), all_preds, all_labels

def save_misclassified_samples(model, loader, device, classes, save_dir, num_samples=20):
    model.eval()
    misclassified = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
            _, predicted = outputs.max(1)
            mask = (predicted != labels)
            if mask.any():
                imgs, lbls, preds = images[mask].cpu(), labels[mask].cpu(), predicted[mask].cpu()
                for i in range(len(imgs)):
                    misclassified.append((imgs[i], lbls[i], preds[i]))
                    if len(misclassified) >= num_samples: break
            if len(misclassified) >= num_samples: break

    plt.figure(figsize=(15, 6))
    for i, (img, lbl, prd) in enumerate(misclassified[:num_samples]):
        img = img.numpy().transpose((1, 2, 0))
        img = (img * np.array([0.2023, 0.1994, 0.2010])) + np.array([0.4914, 0.4822, 0.4465])
        plt.subplot(2, 10, i + 1)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"T:{classes[lbl]}\nP:{classes[prd]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'misclassified_samples.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--augment', type=str, default='standard', choices=['simple', 'standard', 'mixup', 'cutmix', 'randaug'])
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device(args.device)
    save_dir = os.path.join('runs', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    train_loader, test_loader, classes = get_data_loaders(128, augment_mode=args.augment)
    model = get_model().to(device)
    
    # PyTorch 2.0+ コンパイル
    if hasattr(torch, 'compile'):
        model = torch.compile(model)

    criterion = LabelSmoothingLoss(classes=len(classes), smoothing=args.label_smoothing) if args.label_smoothing > 0 else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(args.epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, args.augment)
        v_loss, v_preds, v_labels = evaluate(model, test_loader, criterion, device)
        v_acc = 100. * sum(np.array(v_preds) == np.array(v_labels)) / len(v_labels)
        scheduler.step()
        for k, v in zip(history.keys(), [t_loss, t_acc, v_loss, v_acc]): history[k].append(v)
        print(f"Epoch [{epoch+1}/{args.epochs}] Val Acc: {v_acc:.2f}%")

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    pd.DataFrame(history).to_csv(os.path.join(save_dir, 'history.csv'), index=False)
    
    # グラフ等の保存
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Acc'); plt.plot(history['val_acc'], label='Val Acc')
    plt.legend(); plt.savefig(os.path.join(save_dir, 'acc_curve.png')); plt.close()
    
    cm = confusion_matrix(v_labels, v_preds)
    plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png')); plt.close()
    
    save_misclassified_samples(model, test_loader, device, classes, save_dir)
    print(f"✅ Training Complete. Saved to {save_dir}")

if __name__ == '__main__':
    main()