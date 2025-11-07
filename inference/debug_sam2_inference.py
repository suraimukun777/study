#!/usr/bin/env python
"""
SAM2推論のデバッグスクリプト
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, '/workspace/POT/POT')
sys.path.insert(0, '/workspace/POT_SAM2_Hybrid')
sys.path.insert(0, '/workspace')

from models.pot_sam2_e2e import POTSAM2EndToEnd
from data import data_voc

def debug_single_image():
    """1枚の画像でSAM2推論をデバッグ"""
    
    # モデルをロード
    print("Loading model...")
    model = POTSAM2EndToEnd(
        sam2_checkpoint='/workspace/POT_SAM2_Hybrid/checkpoints/sam2_hiera_large.pt',
        freeze_sam2=True
    ).cuda()
    
    checkpoint = torch.load('/workspace/POT_SAM2_Hybrid/experiments/e2e_training/best.pth')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print("✅ Model loaded")
    
    # データをロード
    print("\nLoading data...")
    os.chdir('/workspace/POT_SAM2_Hybrid')
    
    dataset = data_voc.VOC12ClsDatasetMSF(
        'data/train_test.txt',
        voc12_root='/workspace/POT_SAM2_Hybrid/VOCdevkit/VOC2012',
        scales=(1.0,)
    )
    
    pack = dataset[0]
    
    # データの型を確認
    img_data = pack['img'][0]
    label_data = pack['label'][0]
    
    if isinstance(img_data, np.ndarray):
        img = torch.from_numpy(img_data).unsqueeze(0).cuda()
    else:
        img = img_data.unsqueeze(0).cuda()
    
    if isinstance(label_data, np.ndarray):
        label = torch.from_numpy(label_data).unsqueeze(0).cuda()
    else:
        label = label_data.unsqueeze(0).cuda()
    
    img_name = pack['name'][0]
    
    print(f"Image: {img_name}")
    print(f"Image shape: {img.shape}")
    print(f"Label shape: {label.shape}")
    
    # CLIP-ES CAMをロード
    cam_path = f'/workspace/POT_SAM2_Hybrid/CLIP_ES_refined_CAM/cams_71/{img_name}.npy'
    cam_dict = np.load(cam_path, allow_pickle=True).item()
    cams_clip = cam_dict["attn_highres"]
    keys = cam_dict["keys"]
    
    print(f"CLIP CAM shape: {cams_clip.shape}")
    print(f"Keys: {keys}")
    
    # labelを準備
    label_for_pot = F.pad(label, (1, 0), 'constant', 1.0)
    label_for_pot = label_for_pot.unsqueeze(-1).unsqueeze(-1)
    
    print(f"Label for POT shape: {label_for_pot.shape}")
    print(f"Valid classes: {torch.where(label_for_pot[0, :, 0, 0] > 0)[0]}")
    
    # 推論
    print("\n" + "="*60)
    print("Running inference...")
    print("="*60)
    
    with torch.no_grad():
        outputs = model(img, label_for_pot, cams_clip, keys, return_loss=False)
    
    pred_mask = outputs['masks'][0].cpu().numpy()
    cam = outputs['cam'][0].cpu().numpy()
    
    print(f"\nPredicted mask shape: {pred_mask.shape}")
    print(f"CAM shape: {cam.shape}")
    print(f"Unique values in mask: {np.unique(pred_mask)}")
    print(f"Mask value counts:")
    for val in np.unique(pred_mask):
        count = (pred_mask == val).sum()
        print(f"  Class {val}: {count} pixels ({count/pred_mask.size*100:.2f}%)")
    
    # Ground truthをロード
    gt_path = f'/workspace/POT_SAM2_Hybrid/VOCdevkit/VOC2012/SegmentationClass/{img_name}.png'
    gt_mask = np.array(Image.open(gt_path))
    
    print(f"\nGround truth shape: {gt_mask.shape}")
    print(f"GT unique values: {np.unique(gt_mask)}")
    
    # 可視化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 元画像
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth
    axes[0, 1].imshow(gt_mask, cmap='tab20')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # 予測マスク
    axes[0, 2].imshow(pred_mask, cmap='tab20')
    axes[0, 2].set_title('Predicted Mask (SAM2)')
    axes[0, 2].axis('off')
    
    # CAM（最大値のクラス）
    cam_argmax = np.argmax(cam, axis=0)
    axes[1, 0].imshow(cam_argmax, cmap='tab20')
    axes[1, 0].set_title('CAM (argmax)')
    axes[1, 0].axis('off')
    
    # 各クラスのCAM
    valid_classes = torch.where(label_for_pot[0, :, 0, 0] > 0)[0].cpu().numpy()
    for i, cls_idx in enumerate(valid_classes[:2]):
        if cls_idx == 0:
            continue
        axes[1, 1+i].imshow(cam[cls_idx], cmap='hot')
        axes[1, 1+i].set_title(f'CAM Class {cls_idx}')
        axes[1, 1+i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/workspace/POT_SAM2_Hybrid/experiments/debug_sam2_inference.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: /workspace/POT_SAM2_Hybrid/experiments/debug_sam2_inference.png")

if __name__ == '__main__':
    debug_single_image()

