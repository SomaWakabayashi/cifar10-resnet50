import numpy as np
import torch

def rand_bbox(size, lam):
    """Cutmix用のバウンディングボックスを生成"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # パッチの中心座標をランダムに決定
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Mixup: 画像とラベルを線形補間する"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, beta=1.0, device='cuda'):
    """Cutmix: 画像の一部を別の画像で置き換える"""
    if beta > 0:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    
    # ターゲットラベルの準備
    y_a = y
    y_b = y[index]

    # バウンディングボックスの生成と適用
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # 実際にパッチを貼った割合からlambdaを再計算
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam

def mix_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/Cutmix用の損失計算（2つのラベルのLossを重み付け和する）"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)