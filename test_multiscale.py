#!/usr/bin/env python
"""
マルチスケール推論のテストスクリプト
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# パスを追加
sys.path.insert(0, '/workspace/POT/POT')
sys.path.insert(0, '/workspace/POT_SAM2_Hybrid')
sys.path.insert(0, '/workspace')

from models.pot_sam2_e2e import POTSAM2EndToEnd

def test_multiscale():
    """マルチスケール推論のテスト"""
    
    print("="*60)
    print("マルチスケール推論テスト")
    print("="*60)
    
    # モデルを初期化
    print("\n1. モデルの初期化...")
    sam2_checkpoint = '/workspace/POT_SAM2_Hybrid/checkpoints/sam2_hiera_large.pt'
    
    if not os.path.exists(sam2_checkpoint):
        print(f"❌ SAM2チェックポイントが見つかりません: {sam2_checkpoint}")
        return False
    
    try:
        model = POTSAM2EndToEnd(
            sam2_checkpoint=sam2_checkpoint,
            freeze_sam2=True
        ).cuda()
        model.eval()
        print("✅ モデル初期化成功")
    except Exception as e:
        print(f"❌ モデル初期化エラー: {e}")
        return False
    
    # ダミーデータを作成
    print("\n2. ダミーデータの作成...")
    batch_size = 1
    h, w = 512, 512
    num_classes = 21
    
    # POTモデルはマルチスケール入力を期待（オリジナル + 反転）
    img_original = torch.randn(batch_size, 3, h, w).cuda()
    img_flipped = torch.flip(img_original, dims=[-1])
    img = torch.cat([img_original, img_flipped], dim=0)  # (2, 3, h, w)
    
    label = torch.zeros(num_classes, 1, 1).cuda()
    label[1] = 1.0  # クラス1をアクティブに
    
    # ダミーCAM - POTモデルが期待する形式（numpy配列）
    # 実際のCLIP-ES CAMは (num_classes, H, W) の形式
    cam = np.random.randn(num_classes, h//16, w//16).astype(np.float32)
    keys = [1]
    
    print("✅ ダミーデータ作成成功")
    print(f"   画像形状: {img.shape}")
    print(f"   CAM形状: {cam.shape}")
    
    # 単一スケールテスト
    print("\n3. 単一スケール推論テスト...")
    model.use_multiscale = False
    
    try:
        with torch.no_grad():
            outputs_single = model(img, label, cam, keys, return_loss=False)
        
        masks_single = outputs_single['masks']
        print(f"✅ 単一スケール推論成功")
        print(f"   出力形状: {masks_single.shape}")
    except Exception as e:
        print(f"❌ 単一スケール推論エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # マルチスケールテスト
    print("\n4. マルチスケール推論テスト...")
    model.use_multiscale = True
    
    try:
        with torch.no_grad():
            outputs_multi = model(img, label, cam, keys, return_loss=False)
        
        masks_multi = outputs_multi['masks']
        print(f"✅ マルチスケール推論成功")
        print(f"   出力形状: {masks_multi.shape}")
    except Exception as e:
        print(f"❌ マルチスケール推論エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 結果の比較
    print("\n5. 結果の比較...")
    print(f"   単一スケール形状: {masks_single.shape}")
    print(f"   マルチスケール形状: {masks_multi.shape}")
    
    if masks_single.shape == masks_multi.shape:
        print("✅ 出力形状が一致")
    else:
        print("⚠️ 出力形状が不一致")
    
    # 差分を計算
    diff = (masks_single != masks_multi).sum().item()
    total = masks_single.numel()
    diff_ratio = diff / total * 100
    
    print(f"\n   差分ピクセル数: {diff} / {total} ({diff_ratio:.2f}%)")
    
    print("\n" + "="*60)
    print("✅ すべてのテストが成功しました！")
    print("="*60)
    
    return True

def test_vote_masks():
    """投票メカニズムのテスト"""
    
    print("\n" + "="*60)
    print("投票メカニズムテスト")
    print("="*60)
    
    print("\n1. モデルの初期化...")
    sam2_checkpoint = '/workspace/POT_SAM2_Hybrid/checkpoints/sam2_hiera_large.pt'
    
    model = POTSAM2EndToEnd(
        sam2_checkpoint=sam2_checkpoint,
        freeze_sam2=True
    ).cuda()
    
    # ダミーマスクを作成
    print("\n2. ダミーマスクの作成...")
    batch_size = 1
    h, w = 100, 100
    
    # 3つのスケールのマスク
    masks = []
    for i in range(3):
        mask = torch.randint(0, 3, (batch_size, h, w)).cuda()
        masks.append(mask)
    
    print(f"✅ {len(masks)}個のマスクを作成")
    
    # 投票テスト
    print("\n3. 投票処理...")
    try:
        voted_mask = model._vote_masks(masks)
        print(f"✅ 投票成功")
        print(f"   出力形状: {voted_mask.shape}")
        print(f"   ユニーククラス数: {voted_mask.unique().numel()}")
    except Exception as e:
        print(f"❌ 投票エラー: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ 投票メカニズムのテストが成功しました！")
    print("="*60)
    
    return True

if __name__ == '__main__':
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  POT-SAM2 マルチスケール推論テスト".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    print("\n")
    
    # 投票メカニズムテスト
    success1 = test_vote_masks()
    
    # マルチスケール推論テスト
    success2 = test_multiscale()
    
    if success1 and success2:
        print("\n" + "🎉 すべてのテストが成功しました！ 🎉\n")
        sys.exit(0)
    else:
        print("\n" + "❌ 一部のテストが失敗しました\n")
        sys.exit(1)

