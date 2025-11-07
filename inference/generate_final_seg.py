#!/usr/bin/env python
"""
POT-SAM2 Hybrid: 統合推論スクリプト
POT CAM生成 → SAM2マスク生成 → 評価を一括実行
"""

import subprocess
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='POT-SAM2 Hybrid full pipeline')
    parser.add_argument('--config', type=str,
                       default='/workspace/POT_SAM2_Hybrid/configs/pot_sam2_voc.yaml',
                       help='Config file')
    parser.add_argument('--strategy', type=str, default='points',
                       choices=['points', 'boxes', 'hybrid'],
                       help='Prompting strategy')
    parser.add_argument('--skip_cam', action='store_true',
                       help='Skip CAM generation (use existing)')
    parser.add_argument('--skip_mask', action='store_true',
                       help='Skip mask generation (use existing)')
    return parser.parse_args()

def run_command(cmd, description):
    """コマンドを実行"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        sys.exit(1)
    
    print(f"\n✅ {description} completed")

def main():
    args = parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          POT-SAM2 Hybrid Full Pipeline                   ║
║                                                          ║
║  Strategy: {args.strategy:43s} ║
╚══════════════════════════════════════════════════════════╝
""")
    
    # Step 1: POT CAM生成
    if not args.skip_cam:
        run_command([
            'python', '/workspace/POT_SAM2_Hybrid/inference/generate_pot_cams.py'
        ], 'Step 1: POT CAM Generation')
    else:
        print("\n⏭️  Skipping CAM generation (using existing)")
    
    # Step 2: SAM2マスク生成
    if not args.skip_mask:
        run_command([
            'python', '/workspace/POT_SAM2_Hybrid/inference/generate_sam2_masks.py',
            '--strategy', args.strategy
        ], f'Step 2: SAM2 Mask Generation ({args.strategy})')
    else:
        print("\n⏭️  Skipping mask generation (using existing)")
    
    # Step 3: 評価
    run_command([
        'python', '/workspace/POT_SAM2_Hybrid/utils/evaluate_miou.py'
    ], 'Step 3: Evaluation (mIoU Calculation)')
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          🎉 Pipeline Completed Successfully! 🎉          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == '__main__':
    main()



