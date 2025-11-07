#!/usr/bin/env python
"""
POT-SAM2 Hybrid Validation Script
Validationセットで推論を実行してmIoUを評価
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# パスを追加
sys.path.insert(0, '/workspace/POT/POT')
sys.path.insert(0, '/workspace/POT_SAM2_Hybrid')
sys.path.insert(0, '/workspace')

from models.pot_sam2_e2e import POTSAM2EndToEnd
from data import data_voc

def parse_args():
    parser = argparse.ArgumentParser(description='POT-SAM2 Validation')
    parser.add_argument('--checkpoint', type=str,
                       default='/workspace/POT_SAM2_Hybrid/experiments/e2e_training/best.pth',
                       help='Model checkpoint')
    parser.add_argument('--data_list', type=str,
                       default='/workspace/POT_SAM2_Hybrid/data/val_voc.txt',
                       help='Validation data list')
    parser.add_argument('--voc_root', type=str,
                       default='/workspace/POT_SAM2_Hybrid/VOCdevkit/VOC2012',
                       help='VOC dataset root')
    parser.add_argument('--clip_cam_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/CLIP_ES_refined_CAM/cams_71',
                       help='CLIP-ES CAM directory')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/POT_SAM2_Hybrid/experiments/validation_results',
                       help='Output directory')
    parser.add_argument('--sam2_checkpoint', type=str,
                       default='/workspace/POT_SAM2_Hybrid/checkpoints/sam2_hiera_large.pt',
                       help='SAM2 checkpoint')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of validation samples (0=all)')
    parser.add_argument('--save_vis', action='store_true',
                       help='Save visualization')
    parser.add_argument('--use_multiscale', action='store_true',
                       help='Use multiscale inference (improves small objects)')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use test-time augmentation (TTA) for ensemble')
    
    return parser.parse_args()

def load_model(args):
    """モデルをロード"""
    print("Loading model...")
    
    model = POTSAM2EndToEnd(
        sam2_checkpoint=args.sam2_checkpoint,
        freeze_sam2=True
    ).cuda()
    
    # チェックポイントをロード
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cuda')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✅ Checkpoint loaded")
    else:
        print(f"⚠️ Checkpoint not found: {args.checkpoint}")
        print("Using initialized model")
    
    model.eval()
    return model

def create_dataloader(args):
    """データローダーを作成"""
    print("Loading dataset...")
    
    # POTのdataディレクトリに移動
    original_dir = os.getcwd()
    os.chdir('/workspace/POT_SAM2_Hybrid')
    
    try:
        # データリストを読み込み
        with open(args.data_list, 'r') as f:
            lines = f.readlines()
        
        if args.num_samples > 0:
            lines = lines[:args.num_samples]
            print(f"Using {len(lines)} samples for validation")
        else:
            print(f"Using all {len(lines)} samples for validation")
        
        # 一時ファイルを作成
        temp_list = '/tmp/val_list_temp.txt'
        with open(temp_list, 'w') as f:
            f.writelines(lines)
        
        dataset = data_voc.VOC12ClsDatasetMSF(
            temp_list,
            voc12_root=args.voc_root,
            scales=(1.0,)
        )
    finally:
        os.chdir(original_dir)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader

def compute_iou(pred_mask, gt_mask, num_classes=21):
    """IoUを計算"""
    ious = []
    
    for cls in range(num_classes):
        if cls == 255:  # ignore label
            continue
        
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        intersection = (pred_cls & gt_cls).sum().item()
        union = (pred_cls | gt_cls).sum().item()
        
        if union == 0:
            if intersection == 0:
                ious.append(float('nan'))  # クラスが存在しない
            else:
                ious.append(0.0)
        else:
            ious.append(intersection / union)
    
    return ious

def run_validation(model, dataloader, args):
    """Validation実行"""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)
    
    if args.save_vis:
        os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    all_ious = []
    class_ious = [[] for _ in range(21)]
    
    print("\n" + "="*60)
    print("Running validation...")
    if args.use_multiscale:
        print("🔍 Using multiscale inference (scales: 0.5, 1.0, 1.5)")
    else:
        print("Using single-scale inference")
    if args.use_tta:
        print("🔄 Using test-time augmentation (TTA)")
    print("="*60 + "\n")
    
    # マルチスケール設定をモデルに反映
    model.use_multiscale = args.use_multiscale
    
    with torch.no_grad():
        for idx, pack in enumerate(tqdm(dataloader, desc="Validation")):
            img = pack['img'][0].cuda()
            # labelは元々20次元のベクトル（クラスごとの存在フラグ）
            # pack['label']は(batch, 20)の形式
            label = pack['label'].cuda()  # [0]を削除
            img_name = pack['name'][0]
            
            # CLIP-ES CAMをロード
            cam_path = os.path.join(args.clip_cam_dir, img_name + '.npy')
            if not os.path.exists(cam_path):
                print(f"⚠️ CAM not found: {img_name}")
                continue
            
            cam_dict = np.load(cam_path, allow_pickle=True).item()
            cams_clip = cam_dict["attn_highres"]
            keys = cam_dict["keys"]
            
            # labelを準備
            # labelは(1, 20)の形式（20クラス、背景なし）
            # POTは(21, 1, 1)を期待（21クラス、背景あり、バッチ次元なし）
            label_for_pot = F.pad(label[0], (1, 0), 'constant', 1.0)  # (21,)
            label_for_pot = label_for_pot.unsqueeze(-1).unsqueeze(-1)  # (21, 1, 1)
            
            # 推論
            try:
                # return_loss=Falseで推論モード（SAM2を使用）
                if args.use_tta:
                    # TTA使用
                    outputs = model.test_time_augmentation(img, label_for_pot, cams_clip, keys)
                else:
                    # 通常推論
                    outputs = model(img, label_for_pot, cams_clip, keys, return_loss=False)
                
                # SAM2で生成されたマスクを取得
                pred_mask = outputs['masks'][0].cpu().numpy().astype(np.uint8)
                
            except Exception as e:
                import traceback
                print(f"⚠️ Error processing {img_name}: {e}")
                traceback.print_exc()
                continue
            
            # Ground truthマスクをロード
            gt_path = os.path.join(args.voc_root, 'SegmentationClass', img_name + '.png')
            if os.path.exists(gt_path):
                gt_mask = np.array(Image.open(gt_path))
                
                # マスクをGTと同じサイズにリサイズ
                if pred_mask.shape != gt_mask.shape:
                    pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
                    pred_mask_pil = pred_mask_pil.resize(gt_mask.shape[::-1], Image.NEAREST)
                    pred_mask = np.array(pred_mask_pil)
                
                # IoU計算
                ious = compute_iou(pred_mask, gt_mask, num_classes=21)
                all_ious.append(ious)
                
                for cls, iou in enumerate(ious):
                    if not np.isnan(iou):
                        class_ious[cls].append(iou)
                
                # マスクを保存
                mask_save_path = os.path.join(args.output_dir, 'masks', img_name + '.png')
                Image.fromarray(pred_mask.astype(np.uint8)).save(mask_save_path)
            else:
                print(f"⚠️ GT mask not found: {img_name}")
    
    # 結果を集計
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60 + "\n")
    
    # クラス別IoU
    voc_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    print("Class-wise IoU:")
    print("-" * 60)
    
    valid_class_ious = []
    for cls, class_name in enumerate(voc_classes):
        if len(class_ious[cls]) > 0:
            mean_iou = np.mean(class_ious[cls])
            valid_class_ious.append(mean_iou)
            print(f"{class_name:15s}: {mean_iou:.4f} ({len(class_ious[cls])} samples)")
        else:
            print(f"{class_name:15s}: N/A")
    
    # mIoU
    if len(valid_class_ious) > 0:
        miou = np.mean(valid_class_ious)
        print("\n" + "="*60)
        print(f"mIoU: {miou:.4f} ({miou*100:.2f}%)")
        print("="*60 + "\n")
        
        # 結果をファイルに保存
        result_path = os.path.join(args.output_dir, 'results.txt')
        with open(result_path, 'w') as f:
            f.write("POT-SAM2 Hybrid Validation Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"mIoU: {miou:.4f} ({miou*100:.2f}%)\n\n")
            f.write("Class-wise IoU:\n")
            f.write("-"*60 + "\n")
            for cls, class_name in enumerate(voc_classes):
                if len(class_ious[cls]) > 0:
                    mean_iou = np.mean(class_ious[cls])
                    f.write(f"{class_name:15s}: {mean_iou:.4f}\n")
        
        print(f"✅ Results saved to: {result_path}")
    else:
        print("⚠️ No valid results")

def main():
    args = parse_args()
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "     POT-SAM2 Hybrid Validation".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # モデルをロード
    model = load_model(args)
    
    # データローダーを作成
    dataloader = create_dataloader(args)
    
    # Validation実行
    run_validation(model, dataloader, args)
    
    print("\n✅ Validation completed!")

if __name__ == '__main__':
    main()

