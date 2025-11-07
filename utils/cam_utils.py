"""CAM処理ユーティリティ"""
import numpy as np
import torch
import torch.nn.functional as F

def normalize_cam(cam):
    """CAMを正規化"""
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam

def apply_threshold(cam, threshold=0.5):
    """閾値適用"""
    return (cam > threshold).astype(np.uint8)

def resize_cam(cam, size):
    """CAMをリサイズ"""
    if isinstance(cam, np.ndarray):
        cam = torch.from_numpy(cam)
    
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=size,
        mode='bilinear',
        align_corners=False
    )
    return cam.squeeze().numpy()



