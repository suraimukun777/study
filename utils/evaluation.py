"""評価関数"""
import numpy as np

def compute_iou(pred, gt, num_classes):
    """IoUを計算"""
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    valid_mask = (gt != 255)
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
    intersection = np.diag(confusion_matrix)
    union = (confusion_matrix.sum(axis=1) + 
             confusion_matrix.sum(axis=0) - 
             intersection)
    
    iou = np.zeros(len(intersection))
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    
    return iou



