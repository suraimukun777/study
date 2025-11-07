#!/usr/bin/env python
"""
POT-SAM2 Hybrid: SAM2マスク生成スクリプト
POT CAMからプロンプトを抽出してSAM2でマスクを生成します
"""

import torch
import numpy as np
import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings("ignore")

# SAM2のパスを追加
sys.path.insert(0, '/workspace')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def parse_args():
    parser = argparse.ArgumentParser(description='Generate SAM2 masks from POT CAMs')
    parser.add_argument('--pot_cam_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/experiments/pot_cams',
                       help='POT CAM directory')
    parser.add_argument('--image_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/VOCdevkit/VOC2012/JPEGImages',
                       help='Image directory')
    parser.add_argument('--data_list', type=str,
                       default='/workspace/POT_SAM2_Hybrid/data/val_voc.txt',
                       help='Data list file')
    parser.add_argument('--sam2_checkpoint', type=str,
                       default='/workspace/POT_SAM2_Hybrid/checkpoints/sam2_hiera_large.pt',
                       help='SAM2 checkpoint path')
    parser.add_argument('--sam2_config', type=str,
                       default='sam2_hiera_l.yaml',
                       help='SAM2 config file')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/experiments/sam2_masks',
                       help='Output directory')
    parser.add_argument('--strategy', type=str, default='points',
                       choices=['points', 'boxes', 'hybrid'],
                       help='Prompting strategy')
    parser.add_argument('--num_points', type=int, default=5,
                       help='Number of points per class')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='CAM threshold for point extraction')
    return parser.parse_args()

def load_sam2(args):
    """SAM2モデルをロード"""
    print(f"Loading SAM2 from {args.sam2_checkpoint}")
    sam2 = build_sam2(args.sam2_config, args.sam2_checkpoint, device='cuda')
    predictor = SAM2ImagePredictor(sam2)
    print("✅ SAM2 loaded successfully")
    return predictor

def extract_points_from_cam(cam, num_points=5, threshold=0.5):
    """CAMから点プロンプトを抽出"""
    # 閾値以上の領域を取得
    mask = (cam > threshold).astype(np.uint8)
    
    if mask.sum() == 0:
        # マスクが空の場合、最大値の位置を返す
        y, x = np.unravel_index(cam.argmax(), cam.shape)
        return np.array([[x, y]]), np.array([1])
    
    # 距離変換で中心部を見つける
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    points = []
    labels = []
    
    for _ in range(num_points):
        if dist_transform.max() == 0:
            break
        
        # 最大値の位置を取得
        y, x = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
        points.append([x, y])
        labels.append(1)  # foreground
        
        # 周辺を抑制
        cv2.circle(dist_transform, (x, y), 20, 0, -1)
    
    if len(points) == 0:
        y, x = np.unravel_index(cam.argmax(), cam.shape)
        points.append([x, y])
        labels.append(1)
    
    return np.array(points), np.array(labels)

def extract_box_from_cam(cam, threshold=0.5):
    """CAMからボックスプロンプトを抽出"""
    mask = (cam > threshold).astype(np.uint8)
    
    if mask.sum() == 0:
        # マスクが空の場合、画像全体のボックスを返す
        h, w = cam.shape
        return np.array([0, 0, w, h])
    
    # 輪郭を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        h, w = cam.shape
        return np.array([0, 0, w, h])
    
    # 最大の輪郭のバウンディングボックスを取得
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return np.array([x, y, x + w, y + h])

def generate_masks(args):
    """SAM2でマスクを生成"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # SAM2をロード
    predictor = load_sam2(args)
    
    # データリストを読み込み
    with open(args.data_list, 'r') as f:
        image_names = [line.strip() for line in f.readlines()]
    
    print(f"Generating SAM2 masks for {len(image_names)} images...")
    print(f"Strategy: {args.strategy}")
    
    for img_name in tqdm(image_names, desc="Generating masks"):
        # POT CAMをロード
        cam_path = os.path.join(args.pot_cam_dir, img_name + '.npy')
        if not os.path.exists(cam_path):
            print(f"Warning: CAM not found: {cam_path}, skipping")
            continue
        
        cam_data = np.load(cam_path, allow_pickle=True).item()
        cams = cam_data['cam']  # shape: (num_classes, H, W)
        keys = cam_data['keys']  # クラスインデックス
        
        # 画像をロード
        img_path = os.path.join(args.image_dir, img_name + '.jpg')
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # SAM2に画像を設定
        predictor.set_image(image)
        
        # クラスごとにマスクを生成
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for idx, class_idx in enumerate(keys):
            if class_idx == 0:  # background skip
                continue
            
            cam = cams[idx]
            
            if args.strategy == 'points':
                # 点プロンプト
                points, point_labels = extract_points_from_cam(
                    cam, num_points=args.num_points, threshold=args.threshold
                )
                
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    multimask_output=True
                )
                
            elif args.strategy == 'boxes':
                # ボックスプロンプト
                box = extract_box_from_cam(cam, threshold=args.threshold)
                
                masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=True
                )
                
            elif args.strategy == 'hybrid':
                # ハイブリッド（点+ボックス）
                points, point_labels = extract_points_from_cam(
                    cam, num_points=args.num_points, threshold=args.threshold
                )
                box = extract_box_from_cam(cam, threshold=args.threshold)
                
                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True
                )
            
            # 最もスコアが高いマスクを選択
            best_mask = masks[scores.argmax()]
            
            # final_maskに追加（クラスインデックスで上書き）
            final_mask[best_mask > 0] = class_idx
        
        # 保存
        output_path = os.path.join(args.output_dir, img_name + '.png')
        Image.fromarray(final_mask).save(output_path)
    
    print(f"✅ Mask generation completed. Saved to {args.output_dir}")

if __name__ == '__main__':
    args = parse_args()
    generate_masks(args)

