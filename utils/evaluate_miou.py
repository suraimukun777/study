#!/usr/bin/env python
"""
POT-SAM2 Hybrid: 評価スクリプト
生成されたマスクのmIoUを計算します
"""

import numpy as np
import os
import argparse
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# VOC2012のクラス名
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation mIoU')
    parser.add_argument('--pred_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/experiments/sam2_masks',
                       help='Prediction directory')
    parser.add_argument('--gt_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/VOCdevkit/VOC2012/SegmentationClass',
                       help='Ground truth directory')
    parser.add_argument('--data_list', type=str,
                       default='/workspace/POT_SAM2_Hybrid/data/val_voc.txt',
                       help='Data list file')
    parser.add_argument('--num_classes', type=int, default=21,
                       help='Number of classes (including background)')
    return parser.parse_args()

def compute_iou(pred, gt, num_classes):
    """IoUを計算"""
    # 混同行列を初期化
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # 有効なピクセル（255以外）のマスク
    valid_mask = (gt != 255)
    
    # 混同行列を更新
    pred_valid = pred[valid_mask]
    gt_valid = gt[valid_mask]
    
    for pred_cls in range(num_classes):
        for gt_cls in range(num_classes):
            confusion_matrix[pred_cls, gt_cls] = np.sum(
                (pred_valid == pred_cls) & (gt_valid == gt_cls)
            )
    
    return confusion_matrix

def calculate_miou(confusion_matrix):
    """混同行列からmIoUを計算"""
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(confusion_matrix)
    union = (confusion_matrix.sum(axis=1) + 
             confusion_matrix.sum(axis=0) - 
             intersection)
    
    # ゼロ除算を避ける
    iou = np.zeros(len(intersection))
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    
    return iou

def evaluate(args):
    """評価を実行"""
    # データリストを読み込み
    with open(args.data_list, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]
    
    print(f"Evaluating {len(image_names)} images...")
    print(f"Prediction dir: {args.pred_dir}")
    print(f"Ground truth dir: {args.gt_dir}")
    print("-" * 60)
    
    # 全体の混同行列
    total_confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    
    missing_count = 0
    
    for img_name in tqdm(image_names, desc="Evaluating"):
        # 予測マスクをロード
        pred_path = os.path.join(args.pred_dir, img_name + '.png')
        if not os.path.exists(pred_path):
            missing_count += 1
            continue
        
        pred = np.array(Image.open(pred_path))
        
        # Ground truthをロード
        gt_path = os.path.join(args.gt_dir, img_name + '.png')
        gt = np.array(Image.open(gt_path))
        
        # サイズが異なる場合はリサイズ
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST
            ))
        
        # IoUを計算
        confusion = compute_iou(pred, gt, args.num_classes)
        total_confusion += confusion
    
    if missing_count > 0:
        print(f"\nWarning: {missing_count} predictions not found")
    
    # mIoUを計算
    iou_per_class = calculate_miou(total_confusion)
    
    # 結果を表示
    print("\n" + "=" * 60)
    print("Per-class IoU:")
    print("=" * 60)
    
    for i, class_name in enumerate(VOC_CLASSES):
        if i == 0:  # background
            continue
        print(f"{class_name:15s}: {iou_per_class[i]*100:5.2f}%")
    
    # mIoU（backgroundを除く）
    miou = np.mean(iou_per_class[1:])
    
    print("=" * 60)
    print(f"mIoU (excluding background): {miou*100:.2f}%")
    print("=" * 60)
    
    # 詳細統計
    print(f"\nTotal pixels evaluated: {total_confusion.sum():,}")
    print(f"Images evaluated: {len(image_names) - missing_count}")
    
    # クラスごとの統計
    print("\nClass-wise statistics:")
    print("-" * 60)
    print(f"{'Class':<15} {'IoU':>8} {'TP':>10} {'FP':>10} {'FN':>10}")
    print("-" * 60)
    
    for i in range(1, args.num_classes):
        tp = total_confusion[i, i]
        fp = total_confusion[i, :].sum() - tp
        fn = total_confusion[:, i].sum() - tp
        print(f"{VOC_CLASSES[i]:<15} {iou_per_class[i]*100:7.2f}% {tp:10,} {fp:10,} {fn:10,}")
    
    print("-" * 60)
    
    return miou

if __name__ == '__main__':
    args = parse_args()
    miou = evaluate(args)

