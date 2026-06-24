import torch
import numpy as np
import os
import subprocess
import yaml
from gaussian_renderer import render, network_gui, instanced_render
from utils.image_utils import psnr
import sys
from scene import InstScene
from scene.inst_gaussian_model import InstGaussianModel
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as TF
import time
from torchmetrics.functional import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from torchvision import transforms
from lpips import LPIPS  # pip install lpips

SCENE_NAME = "figurines"

def pad_to_even(img: torch.Tensor):
    """Pad a CHW tensor image to even height and width."""
    c, h, w = img.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return img
    pad = [0, 0, pad_w, pad_h]  # left, top, right, bottom
    img = TF.pad(img, pad, fill=1)  # fill=1 for white, 0 for black
    return img


def tensor_mem_MB(t: torch.Tensor):
    return t.numel() * t.element_size() / 1024 ** 2

# def compress(tensor: torch.Tensor, threshold: float):
#     if torch.all(tensor == 0) or torch.all(tensor.abs() < threshold):
#         print("All zero tensor, no need to compress.")
        
#     flattened = tensor.view(tensor.shape[0], -1)
#     mask = ~(flattened == 0).all(dim=1) & ~(flattened.abs() < threshold).all(dim=1)
#     indices = mask.nonzero(as_tuple=True)[0].to(torch.int)  # int32, 4 bytes
#     values = tensor[mask]  # float32, 4 bytes

#     original_size = sizeof_tensor(tensor)
#     sparse_size = sizeof_tensor(indices) + sizeof_tensor(values)

#     print(
#         f"Sparse size: {sparse_size/1024:.2f} KB, original size: {original_size/1024:.2f} KB, ratio: {sparse_size/original_size:.4f}"
#     )
    

if __name__ == "__main__":

    model_dict = torch.load(
        f"./output/{SCENE_NAME}/chkpnt10000.pth", weights_only=False
    )
    all_template_gs = []
    for k, model in model_dict.items():
        template_gs = InstGaussianModel(sh_degree=3)
        template_gs.restore(model_args=model, training_args=None)

        template_gs.save_ply(
            f"./output/{SCENE_NAME}/inst_gs_{template_gs.template_id}_loaded.ply",
            instancing=False)
        offset_point_clouds = template_gs.get_offset_point_clouds()
        for i, pc in enumerate(offset_point_clouds):
            offset_gs = InstGaussianModel(sh_degree=3)
            offset_gs.create_from_offsets(pc)
            offset_gs.save_ply(
                f"./output/{SCENE_NAME}/offset_inst_gs_{template_gs.template_id}_{i}.ply",
                instancing=False)


        # for idx in range(template_gs.instances_num):
            # print(f"-------------Template {k}, Instance {idx}-------------")
            # print("xyz_offset")
            # xyz_offset = template_gs._xyz_offsets[idx]
            # compress(xyz_offset, threshold=1e-8)

            # print("scaling_offset")
            # scaling_offset = template_gs._scaling_offsets[idx]
            # compress(scaling_offset, threshold=1e-8)

            # print("rotation_offset")
            # rotation_offset = template_gs._rotation_offsets[idx]
            # compress(rotation_offset, threshold=1e-8)

            # print("feature_dc_offset")
            # feature_dc_offset = template_gs._features_dc_offsets[idx]
            # compress(feature_dc_offset, threshold=1e-8)

            # print("feature_rest_offset")
            # feature_rest_offset = template_gs._features_rest_offsets[idx]
            # compress(feature_rest_offset, threshold=1e-8)
            
            # print("opacity_offset")
            # opacity_offset = template_gs._opacity_offsets[idx]
            # compress(opacity_offset, threshold=1e-8)
        all_template_gs.append(template_gs)

    bg_gaussians = GaussianModel(sh_degree=3)
    bg_gaussians.load_ply(f"./data/{SCENE_NAME}/seg_inst/bg.ply")
    # bg_gaussians = None

    parser = ArgumentParser(description="Rendering script for Splat-n-Replace")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.source_path = f"./data/{SCENE_NAME}"
    args.model_path = f"./output/{SCENE_NAME}"

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

    scene = InstScene(lp.extract(args), all_template_gs, shuffle=False)
    cameras = scene.getTrainCameras()

    mem_total = 0
    for temp_gs in all_template_gs:
        temp_gs.instancing()
        mem_total += sum([
            tensor_mem_MB(temp_gs.get_full_xyz),
            tensor_mem_MB(temp_gs.get_full_scaling),
            tensor_mem_MB(temp_gs.get_full_rotation),
            tensor_mem_MB(temp_gs.get_full_opacity),
            tensor_mem_MB(temp_gs.get_full_features),
            tensor_mem_MB(temp_gs.get_xyz),
            tensor_mem_MB(temp_gs.get_scaling),
            tensor_mem_MB(temp_gs.get_rotation),
            tensor_mem_MB(temp_gs.get_opacity),
            2 * tensor_mem_MB(temp_gs.get_features_dc),
            2 * tensor_mem_MB(temp_gs.get_features_rest),
            # tensor_mem_MB(temp_gs._features_dc_offsets),
            # tensor_mem_MB(temp_gs._features_rest_offsets),
        ])
        temp_gs.save_ply(
            f"./output/{SCENE_NAME}/inst_gs_{temp_gs.template_id}.ply",
            instancing=True,
        )

    # mem_total += sum([
    #     tensor_mem_MB(bg_gaussians.get_xyz),
    #     tensor_mem_MB(bg_gaussians.get_scaling),
    #     tensor_mem_MB(bg_gaussians.get_rotation),  
    #     tensor_mem_MB(bg_gaussians.get_opacity),
    #     tensor_mem_MB(bg_gaussians.get_features_dc),
    #     tensor_mem_MB(bg_gaussians.get_features_rest),
    # ])
    print(f"Estimated Gaussian data size: {mem_total:.2f} MB")

    save_path = f"./output/{SCENE_NAME}/rendered_images"
    os.makedirs(save_path, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_mem = torch.cuda.memory_allocated() / 1024**2

    # render_img = instanced_render(
    #     cameras[0], all_template_gs, None, pp.extract(args), background
    # )["render"]
    
    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated() / 1024**2
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Start: {start_mem: .2f} MB")
    print(f"End: {end_mem: .2f} MB")
    print(f"Peak: {peak_mem: .2f} MB")

    lpips_fn = LPIPS(net='vgg').cuda()

    for idx, view in enumerate(cameras):
        torch.cuda.synchronize()

        start_time = time.time()
        render_img = instanced_render(
            view, all_template_gs, bg_gaussians, pp.extract(args), background
        )["render"]
        torch.cuda.synchronize()
        render_time = time.time() - start_time
        fps = 1.0 / render_time

        # IMAGE METRICS
        ground_truth = view.original_image.clamp(0, 1).cuda().unsqueeze(0) 
        render_img = render_img.clamp(0, 1).cuda().unsqueeze(0) 
        psnr = psnr_fn(render_img, ground_truth, data_range=1.0).item()
        ssim = 0
        lpips = 0
        ssim = ssim_fn(render_img, ground_truth, data_range=1.0).item()
        lpips = lpips_fn(render_img*2-1, ground_truth*2-1).mean().item()  # LPIPS 期望输入范围 [-1,1]

        print(f"[{idx}] Render: {render_time*1000:.2f} ms | FPS: {fps:.2f} | "
                f"PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | LPIPS: {lpips:.4f}")
        
        # render_img = pad_to_even(render_img)
        save_image(
            render_img,
            f"{save_path}/{idx}.jpg",
        )

    # # 合成视频（假设图片命名为 0.jpg, 1.jpg, ...）
    # video_path = f"{save_path}/output.mp4"
    # fps = 24  # 帧率，可根据需要调整
    # subprocess.run([
    #     "ffmpeg",
    #     "-y",
    #     "-framerate", str(fps),
    #     "-i", f"{save_path}/%d.jpg",
    #     video_path
    # ])
    # print(f"Video saved to {video_path}")

    # All done
    print("\nRender complete.")
