import torch
import numpy as np
import os
from gaussian_renderer import render, network_gui, instanced_render, instanced_render_cuda
import sys
from scene import InstScene
from scene.cameras import Camera
from scene.inst_gaussian_model import InstGaussianModel
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
import time
from torchmetrics.functional import peak_signal_noise_ratio as psnr_fn
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from torchvision import transforms
from lpips import LPIPS  # pip install lpips

SCENE_NAME = "instancing"

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

def generate_random_transforms(num=500, xyz_range=1.0, min_dist=0.8, scale=1.0):
    """
    返回:
        List[torch.Tensor], 长度为 num
        每个元素是 (4, 4) 的变换矩阵
    """
    transforms = []
    positions = []

    max_trials = num * 100
    trials = 0

    while len(transforms) < num and trials < max_trials:
        trials += 1

        # xoz 平面均匀采样
        tx, tz = (np.random.rand(2) * 2 - 1) * xyz_range

        # 保证最小间距
        if len(positions) > 0:
            prev = np.asarray(positions, dtype=np.float32)
            d = np.linalg.norm(prev - np.array([tx, tz], dtype=np.float32), axis=1)
            if np.any(d < min_dist):
                continue

        positions.append([tx, tz])

        # 随机绕 y 轴旋转
        theta = np.random.rand() * 2 * np.pi
        c, s = np.cos(theta), np.sin(theta)

        T = np.eye(4, dtype=np.float32)

        T[:3, :3] = scale * np.array([
            [ c, 0,  s],
            [ 0, 1,  0],
            [-s, 0,  c],
        ], dtype=np.float32)

        T[:3, 3] = [tx, 0.0, tz]

        transforms.append(torch.tensor(T.T, device="cuda"))

    if len(transforms) < num:
        print(
            f"Warning: only generated {len(transforms)} transforms. "
            f"Try smaller min_dist or larger xyz_range."
        )

    return transforms

def generate_orbit_cameras(
    num=60,
    radius=8.0,
    height=2.0,
    target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
):
    """
    生成一组环绕 target 的相机

    返回:
        List[(R, T)]
        满足 x_cam = R @ x_world + T
    """

    cameras = []
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    for i in range(num):
        theta = 2.0 * np.pi * i / num

        # 相机中心（世界坐标）
        C = np.array([
            radius * np.cos(theta),
            height,
            radius * np.sin(theta)
        ], dtype=np.float32)

        # 相机看向目标
        forward = target - C
        forward = forward / np.linalg.norm(forward)

        # camera x-axis
        right = np.cross(world_up, forward)
        right = right / np.linalg.norm(right)

        # camera y-axis
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)

        # world -> camera rotation
        R = np.stack([right, up, forward], axis=0)

        # world -> camera translation
        T = -R @ C

        cameras.append((R.T.astype(np.float32), T.astype(np.float32)))

    return cameras

# N: 20, 100, 210

if __name__ == "__main__":
    T_list = generate_random_transforms(num=50, xyz_range=8.0, min_dist=0.5, scale=1.0)

    bg_gaussians = None

    parser = ArgumentParser(description="Rendering script for instancing")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])

    args.source_path = f"./data/{SCENE_NAME}"
    args.model_path = f"./output/{SCENE_NAME}"

    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")  

    gpu_save_path = f"./output/{SCENE_NAME}/rendered_images_cuda"
    os.makedirs(gpu_save_path, exist_ok=True)
    cpu_save_path = f"./output/{SCENE_NAME}/rendered_images_cpu"
    os.makedirs(cpu_save_path, exist_ok=True)

    ## Test GPU memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_mem = torch.cuda.memory_allocated() / 1024**2

    with torch.no_grad():
        template_gs = InstGaussianModel(sh_degree=0)
        template_gs.load_ply(f"./output/{SCENE_NAME}/stitch.ply")
        template_gs.set_transforms(T_list)

        ## Test CPU memory
        mem_total = 0
        mem_total += sum([
            tensor_mem_MB(template_gs.get_xyz),
            tensor_mem_MB(template_gs.get_scaling),
            tensor_mem_MB(template_gs.get_rotation),
            tensor_mem_MB(template_gs.get_opacity),
            tensor_mem_MB(template_gs.get_features_dc),
            tensor_mem_MB(template_gs.get_features_rest),
            tensor_mem_MB(template_gs.get_transforms),
        ])

        print(f"Estimated CPU Memory: {mem_total:.2f} MB")

        scene = InstScene(lp.extract(args), template_gs, shuffle=False)
        cameras = scene.getTrainCameras()

        lpips_fn = LPIPS(net='vgg').cuda()

        views = generate_orbit_cameras(
            num=len(cameras),
            radius=8.0,
            height=-5.0,
        )

        # GPU Instancing
        for idx, view in enumerate(cameras):

            view.R, view.T = views[idx]
            view.update_transform()

            torch.cuda.synchronize()

            start_time = time.time()
            render_img = instanced_render_cuda(
                view, template_gs, bg_gaussians, pp.extract(args), background
            )["render"]
            torch.cuda.synchronize()
            render_time = time.time() - start_time
            fps = 1.0 / render_time

            print(f"[{idx}] Render: {render_time*1000:.2f} ms | FPS: {fps:.2f} | ")
            save_image(
                render_img,
                f"{gpu_save_path}/{idx}.jpg",
            )

        # # CPU copy (for comparison)
        # for idx, view in enumerate(cameras):

        #     view.R, view.T = views[idx]
        #     view.update_transform()

        #     torch.cuda.synchronize()

        #     template_gs.instancing()

        #     start_time = time.time()
        #     render_img = instanced_render(
        #         view, [template_gs], bg_gaussians, pp.extract(args), background
        #     )["render"]
        #     torch.cuda.synchronize()
        #     render_time = time.time() - start_time
        #     fps = 1.0 / render_time
            
        #     print(f"[{idx}] Render: {render_time*1000:.2f} ms | FPS: {fps:.2f} | ")
        #     save_image(
        #         render_img,
        #         f"{cpu_save_path}/{idx}.jpg",
        #     )
    
    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated() / 1024**2
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Start: {start_mem: .2f} MB")
    print(f"End: {end_mem: .2f} MB")
    print(f"Peak: {peak_mem: .2f} MB")

    # print(f"Total render time for {len(cameras)} views: {avg_render_time:.4f} seconds")
    # avg_render_time /= len(cameras)

    # All done
    print("\nRender complete.")
    # print(f"Average render time: {avg_render_time:.4f} seconds")
