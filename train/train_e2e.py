#!/usr/bin/env python
"""
POT-SAM2 End-to-Endトレーニングスクリプト
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# パスを追加
sys.path.insert(0, '/workspace/POT/POT')
sys.path.insert(0, '/workspace/POT_SAM2_Hybrid')

from models.pot_sam2_e2e import POTSAM2EndToEnd, POTSAM2Loss
from data import data_voc

def parse_args():
    parser = argparse.ArgumentParser(description='POT-SAM2 End-to-End Training')
    parser.add_argument('--data_list', type=str,
                       default='/workspace/POT_SAM2_Hybrid/data/train_voc.txt',
                       help='Training data list')
    parser.add_argument('--voc_root', type=str,
                       default='/workspace/POT_SAM2_Hybrid/VOCdevkit/VOC2012',
                       help='VOC dataset root')
    parser.add_argument('--clip_cam_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/CLIP_ES_refined_CAM/cams_71',
                       help='CLIP-ES CAM directory')
    parser.add_argument('--sam2_checkpoint', type=str,
                       default='/workspace/POT_SAM2_Hybrid/checkpoints/sam2_hiera_large.pt',
                       help='SAM2 checkpoint')
    parser.add_argument('--pot_checkpoint', type=str,
                       default='/workspace/POT_SAM2_Hybrid/checkpoints/pot/best.pth',
                       help='POT checkpoint (for initialization)')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/experiments/e2e_training',
                       help='Output directory')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    
    # Loss weights
    parser.add_argument('--lambda_cam', type=float, default=1.0,
                       help='CAM loss weight')
    parser.add_argument('--lambda_mask', type=float, default=1.0,
                       help='Mask loss weight')
    parser.add_argument('--lambda_consist', type=float, default=0.5,
                       help='Consistency loss weight')
    
    # Other
    parser.add_argument('--freeze_sam2', action='store_true',
                       help='Freeze SAM2 parameters')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    return parser.parse_args()

def create_dataloader(args):
    """データローダーを作成"""
    # POT_SAM2_Hybridのdataディレクトリに移動してデータセットを作成
    original_dir = os.getcwd()
    os.chdir('/workspace/POT_SAM2_Hybrid')
    
    try:
        dataset = data_voc.VOC12ClsDatasetMSF(
            args.data_list,
            voc12_root=args.voc_root,
            scales=(1.0,)
        )
    finally:
        os.chdir(original_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

def train_epoch(model, dataloader, criterion, optimizer, epoch, writer, args):
    """1エポックのトレーニング"""
    model.train()
    
    total_loss = 0
    loss_dict_sum = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
    
    for iter_idx, pack in enumerate(pbar):
        img = pack['img'][0].cuda()  # (B, 3, H, W)
        label = pack['label'][0].cuda()  # (B, C)
        
        # CLIP-ES CAMをロード
        cam_path = os.path.join(args.clip_cam_dir, pack['name'][0] + '.npy')
        if not os.path.exists(cam_path):
            print(f"Warning: CAM not found for {pack['name'][0]}, skipping...")
            continue
        
        cam_dict = np.load(cam_path, allow_pickle=True).item()
        cams_clip = cam_dict["attn_highres"]  # numpy配列のまま（POTモデルが変換する）
        keys = cam_dict["keys"]
        
        # labelは21クラス全体のラベル（POTモデルが期待する形式）
        # backgroundを追加
        label_for_pot = F.pad(label, (1, 0), 'constant', 1.0)  # (21,)
        label_for_pot = label_for_pot.unsqueeze(-1).unsqueeze(-1)  # (21, 1, 1)
        
        # Forward
        optimizer.zero_grad()
        
        outputs = model(img, label_for_pot, cams_clip, keys, return_loss=True)
        
        # 損失計算（元のラベルを使用）
        loss, loss_dict = criterion(outputs, label, gt_mask=None)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 統計
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0) + v.item()
        
        # プログレスバー更新
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cam': f'{loss_dict["cam_loss"].item():.4f}',
            'consist': f'{loss_dict["consistency_loss"].item():.4f}'
        })
        
        # TensorBoard
        global_step = epoch * len(dataloader) + iter_idx
        if iter_idx % 10 == 0:
            writer.add_scalar('Train/loss', loss.item(), global_step)
            for k, v in loss_dict.items():
                writer.add_scalar(f'Train/{k}', v.item(), global_step)
    
    # エポック平均
    avg_loss = total_loss / len(dataloader)
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict_sum.items()}
    
    return avg_loss, avg_loss_dict

def save_checkpoint(model, optimizer, epoch, loss, args, filename='checkpoint.pth'):
    """チェックポイントを保存"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': args
    }
    
    save_path = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, save_path)
    print(f'✅ Checkpoint saved: {save_path}')

def main():
    args = parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          POT-SAM2 End-to-End Training                    ║
║                                                          ║
║  Epochs: {args.epochs:3d}  |  Batch Size: {args.batch_size:2d}  |  LR: {args.lr:.2e}    ║
╚══════════════════════════════════════════════════════════╝
""")
    
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Freeze SAM2: {args.freeze_sam2}")
    
    # データローダー
    print("\nLoading dataset...")
    dataloader = create_dataloader(args)
    print(f"Training samples: {len(dataloader.dataset)}")
    
    # モデル
    print("\nBuilding model...")
    model = POTSAM2EndToEnd(
        sam2_checkpoint=args.sam2_checkpoint,
        freeze_sam2=args.freeze_sam2
    ).to(device)
    
    # POTの重みをロード
    if args.pot_checkpoint:
        print(f"Loading POT weights from {args.pot_checkpoint}")
        pot_weights = torch.load(args.pot_checkpoint, map_location='cpu')
        # POTのチェックポイントは'net'キーに保存されている
        if 'net' in pot_weights:
            pot_weights = pot_weights['net']
        model.pot.load_state_dict(pot_weights, strict=False)
        print("✅ POT weights loaded (partial)")
    
    # 損失関数
    criterion = POTSAM2Loss(
        lambda_cam=args.lambda_cam,
        lambda_mask=args.lambda_mask,
        lambda_consist=args.lambda_consist
    )
    
    # オプティマイザ
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # スケジューラ
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # トレーニングループ
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        avg_loss, loss_dict = train_epoch(
            model, dataloader, criterion, optimizer, epoch, writer, args
        )
        
        # Learning rate update
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # ログ
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        for k, v in loss_dict.items():
            print(f"  {k}: {v:.4f}")
        print(f"  LR: {current_lr:.2e}")
        
        writer.add_scalar('Epoch/loss', avg_loss, epoch)
        writer.add_scalar('Epoch/lr', current_lr, epoch)
        
        # チェックポイント保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, args, 'best.pth')
        
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, args, f'checkpoint_epoch{epoch+1}.pth')
    
    # 最終チェックポイント
    save_checkpoint(model, optimizer, args.epochs-1, avg_loss, args, 'final.pth')
    
    writer.close()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          🎉 Training Completed! 🎉                      ║
║                                                          ║
║  Best Loss: {best_loss:.4f}                                    ║
║  Output Dir: {args.output_dir:40s}║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == '__main__':
    main()

