#!/usr/bin/env python
"""
SAM2統合のデバッグツール
POT-CAM、プロンプト、SAM2出力、最終マスクを可視化
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, '/workspace/POT/POT')
sys.path.insert(0, '/workspace/POT_SAM2_Hybrid')
sys.path.insert(0, '/workspace')

from models.pot_sam2_e2e import POTSAM2EndToEnd
from data import data_voc

def visualize_debug(img_name='2007_000032', checkpoint_path=None):
    """
    1つの画像でデバッグ可視化を実行
    """
    print(f"\n{'='*70}")
    print(f"デバッグ可視化: {img_name}")
    print(f"{'='*70}\n")
    
    # モデルロード
    print("1. モデルをロード...")
    if checkpoint_path is None:
        checkpoint_path = '/workspace/POT_SAM2_Hybrid/experiments/e2e_training_extended/best.pth'
    
    sam2_checkpoint = '/workspace/POT_SAM2_Hybrid/checkpoints/sam2_hiera_large.pt'
    model = POTSAM2EndToEnd(
        sam2_checkpoint=sam2_checkpoint,
        freeze_sam2=True
    ).cuda()
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"✅ チェックポイントロード: {checkpoint_path}")
    
    model.eval()
    
    # データロード
    print("\n2. データをロード...")
    voc_root = '/workspace/POT_SAM2_Hybrid/VOCdevkit/VOC2012'
    clip_cam_dir = '/workspace/POT_SAM2_Hybrid/CLIP_ES_refined_CAM/cams_71'
    
    # 画像
    img_path = os.path.join(voc_root, 'JPEGImages', f'{img_name}.jpg')
    img_pil = Image.open(img_path).convert('RGB')
    img_np = np.array(img_pil)
    
    # 画像を正規化してテンソルに（元画像とflip画像）
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_original = transform(img_pil).unsqueeze(0)
    img_flipped = transform(img_pil.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0)
    img_tensor = torch.cat([img_original, img_flipped], dim=0).cuda()
    
    # CLIP-ES CAM
    cam_path = os.path.join(clip_cam_dir, f'{img_name}.npy')
    cam_dict = np.load(cam_path, allow_pickle=True).item()
    cams_clip = cam_dict["attn_highres"]
    keys = cam_dict["keys"]
    
    # Label（簡易版：keysから作成）
    label = torch.zeros(21, 1, 1).cuda()
    label[0] = 1.0  # background
    for k in keys:
        if k < 21:
            label[k] = 1.0
    
    print(f"✅ 画像サイズ: {img_np.shape}")
    print(f"✅ クラス数: {len(keys)}")
    print(f"✅ クラスID: {keys}")
    
    # 推論（簡易版とSAM2版の両方）
    print("\n3. 推論を実行...")
    
    with torch.no_grad():
        # 簡易版（POT-CAMのargmax）
        outputs_simple = model(img_tensor, label, cams_clip, keys, return_loss=True)
        cam = outputs_simple['cam']
        mask_simple = cam[0].argmax(dim=0).cpu().numpy()
        
        # SAM2推論版
        outputs_sam2 = model(img_tensor, label, cams_clip, keys, return_loss=False)
        mask_sam2 = outputs_sam2['masks'][0].cpu().numpy()
    
    print(f"✅ POT-CAM形状: {cam.shape}")
    print(f"✅ 簡易版マスク形状: {mask_simple.shape}")
    print(f"✅ SAM2版マスク形状: {mask_sam2.shape}")
    
    # Ground Truth
    gt_path = os.path.join(voc_root, 'SegmentationClass', f'{img_name}.png')
    gt_mask = np.array(Image.open(gt_path))
    
    # マスクをGTサイズにリサイズ
    if mask_simple.shape != gt_mask.shape:
        mask_simple = np.array(Image.fromarray(mask_simple.astype(np.uint8)).resize(
            gt_mask.shape[::-1], Image.NEAREST))
        mask_sam2 = np.array(Image.fromarray(mask_sam2.astype(np.uint8)).resize(
            gt_mask.shape[::-1], Image.NEAREST))
    
    # IoU計算
    def compute_miou(pred, gt):
        ious = []
        for cls in range(21):
            pred_cls = (pred == cls)
            gt_cls = (gt == cls)
            intersection = (pred_cls & gt_cls).sum()
            union = (pred_cls | gt_cls).sum()
            if union > 0:
                ious.append(intersection / union)
        return np.mean(ious) if ious else 0
    
    miou_simple = compute_miou(mask_simple, gt_mask)
    miou_sam2 = compute_miou(mask_sam2, gt_mask)
    
    print(f"\n4. 性能比較:")
    print(f"   簡易版 mIoU: {miou_simple:.4f} ({miou_simple*100:.2f}%)")
    print(f"   SAM2版 mIoU: {miou_sam2:.4f} ({miou_sam2*100:.2f}%)")
    print(f"   差分: {(miou_simple - miou_sam2)*100:.2f}ポイント")
    
    # 可視化
    print("\n5. 可視化を作成...")
    
    voc_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # クラスごとに詳細可視化
    n_classes = len(keys)
    fig, axes = plt.subplots(n_classes, 5, figsize=(20, 4*n_classes))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for i, cls_idx in enumerate(keys):
        if cls_idx == 0:  # background skip
            continue
            
        cls_name = voc_classes[cls_idx] if cls_idx < len(voc_classes) else f"class_{cls_idx}"
        
        # 元画像
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'{cls_name}\n元画像')
        axes[i, 0].axis('off')
        
        # POT-CAM
        cam_cls = cam[0, cls_idx].cpu().numpy()
        cam_cls_resized = np.array(Image.fromarray(cam_cls).resize(
            img_np.shape[:2][::-1], Image.BILINEAR))
        axes[i, 1].imshow(img_np)
        axes[i, 1].imshow(cam_cls_resized, alpha=0.5, cmap='jet')
        axes[i, 1].set_title(f'POT-CAM\nmax={cam_cls.max():.3f}')
        axes[i, 1].axis('off')
        
        # プロンプト可視化（ポイント/ボックス）
        axes[i, 2].imshow(img_np)
        axes[i, 2].imshow(cam_cls_resized, alpha=0.3, cmap='jet')
        
        # プロンプト抽出
        cam_area = (cam_cls > 0.3).sum()
        if cam_area > 5000:
            # ボックスプロンプト
            box = model._extract_box_from_cam(cam_cls, threshold=0.5, margin=5)
            if box is not None:
                # ボックスを画像サイズにスケール
                scale_x = img_np.shape[1] / cam_cls.shape[1]
                scale_y = img_np.shape[0] / cam_cls.shape[0]
                x1, y1, x2, y2 = box
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                
                from matplotlib.patches import Rectangle
                rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                linewidth=2, edgecolor='red', facecolor='none')
                axes[i, 2].add_patch(rect)
                axes[i, 2].set_title(f'ボックスプロンプト\narea={cam_area}')
        else:
            # ポイントプロンプト
            num_prototypes = 2 if cam_area < 1000 else 3
            points, labels = model._extract_prototype_centers_from_cam(
                cam_cls, num_prototypes=num_prototypes, threshold=0.3)
            
            if len(points) > 0:
                # ポイントを画像サイズにスケール
                scale_x = img_np.shape[1] / cam_cls.shape[1]
                scale_y = img_np.shape[0] / cam_cls.shape[0]
                points_scaled = points.copy()
                points_scaled[:, 0] *= scale_x
                points_scaled[:, 1] *= scale_y
                
                axes[i, 2].scatter(points_scaled[:, 0], points_scaled[:, 1], 
                                 c='red', s=100, marker='*')
                axes[i, 2].set_title(f'{len(points)}ポイントプロンプト\narea={cam_area}')
        
        axes[i, 2].axis('off')
        
        # 簡易版マスク
        mask_simple_cls = (mask_simple == cls_idx).astype(np.uint8)
        axes[i, 3].imshow(img_np)
        axes[i, 3].imshow(mask_simple_cls, alpha=0.5, cmap='Reds')
        
        # IoU計算
        gt_cls = (gt_mask == cls_idx).astype(np.uint8)
        iou_simple = (mask_simple_cls & gt_cls).sum() / ((mask_simple_cls | gt_cls).sum() + 1e-8)
        axes[i, 3].set_title(f'簡易版マスク\nIoU={iou_simple:.3f}')
        axes[i, 3].axis('off')
        
        # SAM2版マスク
        mask_sam2_cls = (mask_sam2 == cls_idx).astype(np.uint8)
        axes[i, 4].imshow(img_np)
        axes[i, 4].imshow(mask_sam2_cls, alpha=0.5, cmap='Blues')
        
        iou_sam2 = (mask_sam2_cls & gt_cls).sum() / ((mask_sam2_cls | gt_cls).sum() + 1e-8)
        axes[i, 4].set_title(f'SAM2版マスク\nIoU={iou_sam2:.3f}')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    
    # 保存
    output_dir = '/workspace/POT_SAM2_Hybrid/debug_output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{img_name}_debug.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 可視化を保存: {output_path}")
    
    plt.close()
    
    # サマリー
    print(f"\n{'='*70}")
    print("デバッグサマリー")
    print(f"{'='*70}")
    print(f"画像: {img_name}")
    print(f"簡易版 mIoU: {miou_simple:.4f} ({miou_simple*100:.2f}%)")
    print(f"SAM2版 mIoU: {miou_sam2:.4f} ({miou_sam2*100:.2f}%)")
    print(f"性能差: {(miou_simple - miou_sam2)*100:.2f}ポイント")
    print(f"{'='*70}\n")
    
    return {
        'img_name': img_name,
        'miou_simple': miou_simple,
        'miou_sam2': miou_sam2,
        'output_path': output_path
    }

def main():
    """複数の画像でデバッグ"""
    # 代表的な画像を選択
    test_images = [
        '2007_000032',  # 最初のサンプル
        '2007_000039',  # 2番目のサンプル
        '2007_000063',  # 3番目のサンプル
    ]
    
    results = []
    for img_name in test_images:
        try:
            result = visualize_debug(img_name)
            results.append(result)
        except Exception as e:
            print(f"⚠️ エラー ({img_name}): {e}")
            import traceback
            traceback.print_exc()
    
    # 全体サマリー
    if results:
        print(f"\n{'='*70}")
        print("全体サマリー")
        print(f"{'='*70}")
        
        avg_simple = np.mean([r['miou_simple'] for r in results])
        avg_sam2 = np.mean([r['miou_sam2'] for r in results])
        
        print(f"平均 簡易版 mIoU: {avg_simple:.4f} ({avg_simple*100:.2f}%)")
        print(f"平均 SAM2版 mIoU: {avg_sam2:.4f} ({avg_sam2*100:.2f}%)")
        print(f"平均 性能差: {(avg_simple - avg_sam2)*100:.2f}ポイント")
        print(f"{'='*70}\n")

if __name__ == '__main__':
    main()

