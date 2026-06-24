import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import sys
from tqdm import tqdm

def load_images(folder):
    images = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(folder, fname)).convert('RGB')
            images.append(np.array(img))
    return images

def calculate_metrics(pred_folder, gt_folder):
    pred_images = load_images(pred_folder)
    gt_images = load_images(gt_folder)
    assert len(pred_images) == len(gt_images), "Folders must have the same number of images"

    psnr_list, ssim_list, lpips_list = [], [], []
    loss_fn = lpips.LPIPS(net='alex')

    for pred, gt in tqdm(zip(pred_images, gt_images)):
        psnr = peak_signal_noise_ratio(gt, pred, data_range=255)
        ssim = structural_similarity(gt, pred, win_size=3, channel_axis=-1, data_range=255)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        pred_tensor = lpips.im2tensor(pred)
        gt_tensor = lpips.im2tensor(gt)
        lpips_val = loss_fn(pred_tensor, gt_tensor).item()
        lpips_list.append(lpips_val)

    print(f"PSNR: {np.mean(psnr_list):.4f}")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
    print(f"LPIPS: {np.mean(lpips_list):.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluate.py <pred_folder> <gt_folder>")
        exit(1)
    calculate_metrics(sys.argv[1], sys.argv[2])

#     PSNR: 10.7457
#     SSIM: 0.2059
#     LPIPS: 0.6756