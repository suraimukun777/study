#!/usr/bin/env python
"""
SAM2ファインチューニング付きトレーニング
SAM2デコーダの最後の層を解凍して学習
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, '/workspace/POT/POT')
sys.path.insert(0, '/workspace/POT_SAM2_Hybrid')
sys.path.insert(0, '/workspace')

from models.pot_sam2_e2e import POTSAM2EndToEnd, POTSAM2Loss
from data import data_voc

def parse_args():
    parser = argparse.ArgumentParser(description='SAM2ファインチューニング')
    
    # トレーニング設定
    parser.add_argument('--epochs', type=int, default=15, help='エポック数')
    parser.add_argument('--batch_size', type=int, default=1, help='バッチサイズ')
    parser.add_argument('--lr_pot', type=float, default=1e-5, help='POTの学習率')
    parser.add_argument('--lr_sam2', type=float, default=1e-6, help='SAM2の学習率（より低い）')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # SAM2ファインチューニング設定
    parser.add_argument('--unfreeze_sam2_layers', type=int, default=4, 
                        help='SAM2デコーダの解凍層数（デフォルト4層）')
    
    # データ設定
    parser.add_argument('--data_list', type=str, 
                        default='/workspace/POT/POT/data/trainaug_voc.txt',
                        help='トレーニングデータリスト')
    parser.add_argument('--voc_root', type=str,
                        default='/workspace/POT_SAM2_Hybrid/VOCdevkit/VOC2012',
                        help='VOC2012ルートディレクトリ')
    parser.add_argument('--clip_cam_dir', type=str,
                        default='/workspace/POT_SAM2_Hybrid/CLIP_ES_refined_CAM/cams_71',
                        help='CLIP-ES CAMディレクトリ')
    
    # モデル設定
    parser.add_argument('--sam2_checkpoint', type=str,
                        default='/workspace/POT_SAM2_Hybrid/checkpoints/sam2_hiera_large.pt',
                        help='SAM2チェックポイント')
    parser.add_argument('--resume', type=str, default=None,
                        help='再開するチェックポイント')
    
    # 出力設定
    parser.add_argument('--output_dir', type=str,
                        default='experiments/finetune_sam2',
                        help='出力ディレクトリ')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='チェックポイント保存頻度（エポック）')
    
    return parser.parse_args()

def get_optimizer(model, args):
    """
    POTとSAM2で異なる学習率を設定
    """
    # POTのパラメータ
    pot_params = []
    for name, param in model.named_parameters():
        if 'pot' in name and param.requires_grad:
            pot_params.append(param)
    
    # SAM2のパラメータ
    sam2_params = model.get_sam2_trainable_params()
    
    print(f"\n📊 パラメータ設定:")
    print(f"  POT: {sum(p.numel() for p in pot_params):,} パラメータ (lr={args.lr_pot})")
    print(f"  SAM2: {sum(p.numel() for p in sam2_params):,} パラメータ (lr={args.lr_sam2})")
    
    # 異なる学習率で最適化
    optimizer = torch.optim.AdamW([
        {'params': pot_params, 'lr': args.lr_pot},
        {'params': sam2_params, 'lr': args.lr_sam2}
    ], weight_decay=args.weight_decay)
    
    return optimizer

def train_epoch(model, dataloader, criterion, optimizer, epoch, device):
    """
    1エポックのトレーニング
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, pack in enumerate(pbar):
        try:
            # データ準備
            img_name = pack['name'][0]
            img_original = pack['img'].to(device)
            label = pack['label'].to(device)
            
            # POTモデルは元画像とflip画像の両方を期待
            img_flipped = torch.flip(img_original, dims=[-1])
            img = torch.cat([img_original, img_flipped], dim=0)  # (2, 3, H, W)
            
            # CLIP-ES CAM読み込み
            clip_cam_path = os.path.join(args.clip_cam_dir, f'{img_name}.npy')
            if not os.path.exists(clip_cam_path):
                continue
            
            cam_dict = np.load(clip_cam_path, allow_pickle=True).item()
            cams_clip = cam_dict["attn_highres"]
            keys = cam_dict["keys"]
            
            # Forward
            outputs = model(img, label, cams_clip, keys, return_loss=True)
            
            # 損失計算
            losses = criterion(outputs, label)
            loss = losses['total_loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング（安定性向上）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 統計
            total_loss += loss.item()
            num_batches += 1
            
            # 進捗表示
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
            
        except Exception as e:
            print(f"\n⚠️ エラー ({img_name}): {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def main():
    global args
    args = parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("POT-SAM2 ファインチューニング")
    print("="*70)
    print(f"\n設定:")
    print(f"  エポック数: {args.epochs}")
    print(f"  バッチサイズ: {args.batch_size}")
    print(f"  POT学習率: {args.lr_pot}")
    print(f"  SAM2学習率: {args.lr_sam2}")
    print(f"  SAM2解凍層数: {args.unfreeze_sam2_layers}")
    print(f"  出力: {args.output_dir}")
    
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  デバイス: {device}")
    
    # モデル構築
    print("\n📦 モデルをロード...")
    model = POTSAM2EndToEnd(
        sam2_checkpoint=args.sam2_checkpoint,
        freeze_sam2=True,
        unfreeze_sam2_decoder_layers=args.unfreeze_sam2_layers
    ).to(device)
    
    # チェックポイントから再開
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        print(f"\n📂 チェックポイントから再開: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
    
    # 損失関数
    criterion = POTSAM2Loss()
    
    # オプティマイザ
    optimizer = get_optimizer(model, args)
    
    # データローダー
    print("\n📚 データをロード...")
    dataset = data_voc.VOC12ClsDataset(
        args.data_list,
        voc12_root=args.voc_root
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"  データ数: {len(dataset)}")
    print(f"  バッチ数: {len(dataloader)}")
    
    # トレーニングループ
    print("\n🚀 トレーニング開始...\n")
    
    for epoch in range(start_epoch, args.epochs):
        # トレーニング
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, epoch, device)
        
        print(f"\n📊 Epoch {epoch} 完了")
        print(f"  平均損失: {avg_loss:.4f}")
        
        # チェックポイント保存
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_path = os.path.join(args.output_dir, f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, checkpoint_path)
            print(f"  ✅ チェックポイント保存: {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(args.output_dir, 'best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, best_path)
            print(f"  🏆 ベストモデル更新: {best_path} (loss={avg_loss:.4f})")
    
    print("\n" + "="*70)
    print("✅ トレーニング完了！")
    print("="*70)
    print(f"\n最良損失: {best_loss:.4f}")
    print(f"最終チェックポイント: {args.output_dir}/best.pth")

if __name__ == '__main__':
    main()

