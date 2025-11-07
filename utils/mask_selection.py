"""マスク選択ロジック"""
import numpy as np
import torch

def select_best_mask(masks, scores, cam=None):
    """
    複数のマスクから最良のものを選択
    
    Args:
        masks: SAM2が生成したマスク (N, H, W)
        scores: 各マスクのスコア (N,)
        cam: POT-CAM (optional) - CAMとの一貫性を考慮
    
    Returns:
        best_mask: 選択されたマスク
    """
    if cam is not None:
        # CAMとマスクの一貫性スコアを計算
        consistency_scores = []
        for mask in masks:
            # マスク領域内のCAM平均値
            cam_score = (cam * mask).sum() / (mask.sum() + 1e-8)
            consistency_scores.append(cam_score)
        
        consistency_scores = np.array(consistency_scores)
        
        # SAM2スコアとCAM一貫性スコアを組み合わせ
        combined_scores = scores * 0.7 + consistency_scores * 0.3
        best_idx = combined_scores.argmax()
    else:
        # スコアのみで選択
        best_idx = scores.argmax()
    
    return masks[best_idx]

def merge_masks(masks, class_ids):
    """
    複数クラスのマスクを1つのセグメンテーションマップに統合
    
    Args:
        masks: マスクのリスト [(H, W), ...]
        class_ids: 各マスクのクラスID
    
    Returns:
        final_mask: 統合されたマスク (H, W)
    """
    h, w = masks[0].shape
    final_mask = np.zeros((h, w), dtype=np.uint8)
    
    for mask, class_id in zip(masks, class_ids):
        final_mask[mask > 0] = class_id
    
    return final_mask



