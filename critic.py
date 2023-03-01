import numpy as np
from skimage.metrics import *

def calculate_ssim(img1,img2):
    """
        计算 ssim
        img1:Tensor
        img2:Tensor
    """
    img1=np.array(img1).astype(np.float64)*255
    img2=np.array(img2).astype(np.float64)*255
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    ssim_score=structural_similarity(img1,img2,channel_axis=1)
    return ssim_score
    

def calculate_psnr(img1, img2):
    """
        计算 psnr
        img1: Tensor
        img2: Tensor
    """
    # 转为 float64 防止精度丢失
    test1=np.array(img1).astype(np.float64)
    test2=np.array(img2).astype(np.float64)
    return peak_signal_noise_ratio(test1,test2)

