#!/usr/bin/env python
"""
POT-SAM2 Hybrid: POT CAM生成スクリプト
トレーニング済みPOTモデルを使用してCAMを生成します
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# POTのパスを追加
sys.path.insert(0, '/workspace/POT/POT')
from data import data_voc
from tool import pyutils

def parse_args():
    parser = argparse.ArgumentParser(description='Generate POT CAMs')
    parser.add_argument('--model_path', type=str, 
                       default='/workspace/POT/POT/exp/ckpt/best.pth',
                       help='Path to POT model checkpoint')
    parser.add_argument('--network', type=str, default='network.resnet50_POT',
                       help='Network architecture')
    parser.add_argument('--data_list', type=str, 
                       default='/workspace/POT/POT/data/val_voc.txt',
                       help='Path to data list file')
    parser.add_argument('--voc_root', type=str,
                       default='/workspace/POT/POT/VOCdevkit/VOC2012',
                       help='VOC dataset root')
    parser.add_argument('--clip_cam_dir', type=str,
                       default='/workspace/POT/POT/CLIP_ES_refined_CAM/cams_71',
                       help='CLIP-ES CAM directory')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/experiments/pot_cams',
                       help='Output directory for POT CAMs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    return parser.parse_args()

def load_model(args):
    """POTモデルをロード"""
    print(f"Loading POT model from {args.model_path}")
    
    # ネットワークをインポート
    import importlib
    model_module = importlib.import_module(args.network)
    model = model_module.Net()
    
    # チェックポイントをロード
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    model.cuda()
    
    print("✅ Model loaded successfully")
    return model

def generate_cams(args):
    """CAMを生成"""
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モデルをロード
    model = load_model(args)
    
    # データセットを準備
    print(f"Loading dataset from {args.data_list}")
    dataset = data_voc.VOC12ClsDataset(
        args.data_list,
        voc12_root=args.voc_root,
        scales=(1.0,)  # 単一スケール
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Generating CAMs for {len(dataset)} images...")
    
    with torch.no_grad():
        for pack in tqdm(data_loader, desc="Generating CAMs"):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']
            
            # ラベルにbackgroundを追加
            label = F.pad(label, (1, 0), 'constant', 1.0)
            
            # CLIP-ES CAMをロード
            cam_path = os.path.join(args.clip_cam_dir, img_name + '.npy')
            if not os.path.exists(cam_path):
                print(f"Warning: CAM file not found: {cam_path}, skipping")
                continue
            
            cam_dict = np.load(cam_path, allow_pickle=True).item()
            cams_clip = cam_dict["attn_highres"]
            keys = cam_dict["keys"]
            
            # POTモデルで推論
            img = pack['img'][0].cuda(non_blocking=True)
            label_cuda = label.cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)
            
            with torch.cuda.amp.autocast():
                outputs = model(img, label_cuda, cams_clip, keys, pack)
            
            # CAMを取得 (outputs[0]: cam_class, outputs[1]: cam_add)
            cam_class = outputs[0].cpu()
            cam_add = outputs[1].cpu()
            
            # 元の画像サイズにリサイズ
            cam_class = F.interpolate(
                torch.unsqueeze(cam_class.float(), 1),
                size, mode='bilinear', align_corners=False
            )[:, 0]
            
            cam_add = F.interpolate(
                torch.unsqueeze(cam_add, 1),
                size, mode='bilinear', align_corners=False
            )[:, 0]
            
            # 正規化
            cam_class = cam_class / (F.adaptive_max_pool2d(cam_class, (1, 1)) + 1e-5)
            cam_add = cam_add / (F.adaptive_max_pool2d(cam_add, (1, 1)) + 1e-5)
            
            # 2つのCAMを結合
            cam_combined = cam_class + cam_add
            cam_combined = cam_combined / (F.adaptive_max_pool2d(cam_combined, (1, 1)) + 1e-5)
            
            # 有効なクラスのみ保存
            valid_cat = torch.nonzero(label)[:, 0].cpu().numpy()
            cam_combined = cam_combined[valid_cat].numpy()
            
            # 保存
            output_path = os.path.join(args.output_dir, img_name + '.npy')
            np.save(output_path, {
                'cam': cam_combined,
                'keys': valid_cat,
                'size': size
            })
    
    print(f"✅ CAM generation completed. Saved to {args.output_dir}")

if __name__ == '__main__':
    args = parse_args()
    generate_cams(args)

