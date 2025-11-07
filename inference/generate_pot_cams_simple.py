#!/usr/bin/env python
"""
POT-SAM2 Hybrid: 簡易POT CAM生成スクリプト  
CLIP-ES CAMをそのまま使用（chainercv不要）
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate POT CAMs (Simple)')
    parser.add_argument('--data_list', type=str, 
                       default='/workspace/POT_SAM2_Hybrid/data/val_voc.txt',
                       help='Path to data list file')
    parser.add_argument('--voc_root', type=str,
                       default='/workspace/POT_SAM2_Hybrid/VOCdevkit/VOC2012',
                       help='VOC dataset root')
    parser.add_argument('--clip_cam_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/CLIP_ES_refined_CAM/cams_71',
                       help='CLIP-ES CAM directory')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/experiments/pot_cams',
                       help='Output directory')
    return parser.parse_args()

def generate_cams(args):
    """CLIP-ES CAMをリサイズして保存"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 画像リストを読み込み
    with open(args.data_list, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
    
    image_names = [line[0].split('/')[-1].replace('.jpg', '') for line in lines]
    
    print(f"Processing {len(image_names)} CAMs...")
    
    for img_name in tqdm(image_names, desc="Processing CAMs"):
        # CLIP-ES CAMをロード
        cam_path = os.path.join(args.clip_cam_dir, img_name + '.npy')
        if not os.path.exists(cam_path):
            print(f"Warning: CAM not found: {cam_path}, skipping")
            continue
        
        cam_dict = np.load(cam_path, allow_pickle=True).item()
        cams_clip = cam_dict["attn_highres"]
        keys = cam_dict["keys"]
        
        # 画像サイズを取得
        img_path = os.path.join(args.voc_root, 'JPEGImages', img_name + '.jpg')
        img = Image.open(img_path)
        w, h = img.size
        
        # CAMをリサイズ
        cam_tensor = torch.from_numpy(cams_clip).float()
        cam_resized = F.interpolate(
            cam_tensor.unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )[0]
        
        # 正規化
        cam_normalized = cam_resized / (cam_resized.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        
        # 保存
        output_path = os.path.join(args.output_dir, img_name + '.npy')
        np.save(output_path, {
            'cam': cam_normalized.numpy(),
            'keys': keys,
            'size': (h, w)
        })
    
    print(f"✅ CAM processing completed. Saved to {args.output_dir}")

if __name__ == '__main__':
    args = parse_args()
    generate_cams(args)

