import torch
from utils.graphics_utils import geom_transform_quat, geom_transform_points
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui, instanced_render
import sys
from scene import InstScene
from scene.inst_gaussian_model import InstGaussianModel
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
import json

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import torch.nn.functional as F


SCENE_NAME = "hkust-gz"
empty_gaussian_threshold = 100


def masked_interval(mask: torch.Tensor, s: int, e: int):
    new_mask = torch.zeros_like(mask, dtype=torch.bool)
    new_mask[s:e] = mask[s:e]
    return new_mask


def save_image_pair_cv2(pred_img, gt_img, step, save_dir="./tmp"):
    import os

    os.makedirs(save_dir, exist_ok=True)

    pred_np = (
        pred_img.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255
    ).astype("uint8")
    gt_np = (gt_img.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(
        "uint8"
    )

    # OpenCV 是 BGR 格式，所以要转换
    cv2.imwrite(
        f"{save_dir}/pred_{step:05d}.png", cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(f"{save_dir}/gt_{step:05d}.png", cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))


def training(
    dataset,
    opt,
    pipe,
    all_temp_gs,
    bg_gs,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    first_iter = 0
    scene = InstScene(dataset, all_temp_gs)

    for temp_gs in all_temp_gs:
        temp_gs.training_setup(opt)

    if bg_gs is not None:
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        if bg_gs is None:
            background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        iter_start.record()

        for temp_gs in all_temp_gs:
            temp_gs.update_learning_rate(iteration)
            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                temp_gs.oneupSHdegree()

        # Pick a random Camera
        cameras = scene.getTrainCameras()
        if not viewpoint_stack:
            viewpoint_stack = cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # start_time = time.time()
        render_pkg = instanced_render(
            viewpoint_cam, all_temp_gs[0], bg_gs, pipe, background
        )
        image = render_pkg["render"]
        # print("render time", time.time() - start_time)

        gt_image = viewpoint_cam.original_image.cuda()

        # mask = torch.load(
        #     f"{dataset.source_path}/instance_mask/{viewpoint_cam.image_name}.pt"
        # )
        # mask = mask.float().unsqueeze(0).unsqueeze(0)  # [N,C,H,W]
        # mask_resized = (
        #     F.interpolate(
        #         mask, size=(gt_image.shape[1], gt_image.shape[2]), mode="nearest"
        #     )
        #     .squeeze(0)
        #     .bool()
        # )

        # mask3 = mask_resized.expand_as(gt_image)  # 变成 3 通道掩码
        # unmask3 = ~mask_resized.expand_as(gt_image)  # 变成 3 通道掩码
        # masked_gt_image = gt_image * mask3
        # masked_image = image * mask3
        # unmasked_gt_image = gt_image * unmask3
        # unmasked_image = image * unmask3

        # mask_Ll1 = l1_loss(masked_image, masked_gt_image)
        # mask_loss = (1.0 - opt.lambda_dssim) * mask_Ll1 + opt.lambda_dssim * (
        #     1.0 - ssim(masked_image, masked_gt_image)
        # )
        # unmask_Ll1 = l1_loss(unmasked_image, unmasked_gt_image)
        # unmask_loss = (1.0 - opt.lambda_dssim) * unmask_Ll1 + opt.lambda_dssim * (
        #     1.0 - ssim(unmasked_image, unmasked_gt_image)
        # )
        # loss = mask_loss + unmask_loss

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
        )

        total_loss = loss

        total_loss.backward()

        iter_end.record()

        if iteration % 1000 == 0:
            save_image_pair_cv2(
                image,
                gt_image,
                iteration,
                save_dir=f"./output/{SCENE_NAME}/training_images",
            )

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            mse = F.mse_loss(image, gt_image)
            psnr = -10 * torch.log10(mse + 1e-8)  # 避免 log(0)
            progress_bar.set_postfix(
                {"Loss": f"{ema_loss_for_log:.{7}f}", "PSNR": f"{psnr.item():.{2}f}"}
            )
            progress_bar.update()
            if iteration == opt.iterations:
                progress_bar.close()

            # Optimizer step
            if iteration < opt.iterations:
                for temp_gs in all_temp_gs:
                    temp_gs.optimizer.step()
                    temp_gs.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:

                model_dict = {}
                for temp_gs in all_temp_gs:
                    model_dict[temp_gs.template_id] = temp_gs.capture()
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    model_dict,
                    f"./output/{SCENE_NAME}/chkpnt{iteration}.pth",
                )


if __name__ == "__main__":

    with open(f"./data/{SCENE_NAME}/scene_graph.json", "r") as f:
        scene_graph = json.load(f)

    all_template_gs = []

    for temp_map in scene_graph:
        template_id = temp_map["template_id"]
        print(f"Processing template: {template_id}")

        template_gs = InstGaussianModel(sh_degree=3)
        # 1. 加载所有实例，转换到模板空间
        all_instances = []
        transforms = []
        for inst in temp_map["instances"]:
            inst_path = (
                f'./data/{SCENE_NAME}/seg_inst/{inst["instance_id"]}.ply'
            )
            gs = GaussianModel(sh_degree=3)
            gs.load_ply(inst_path)
            # 注意：这里转置了一下，是为了方便与xyz做矩阵乘法
            t = torch.from_numpy(np.array(inst["transform"]).T).float().to("cuda")
            gs._xyz = geom_transform_points(gs.get_xyz, t)
            all_instances.append(gs)
            transforms.append(torch.inverse(t))

        # 2. 合并所有模型为 shared_model（几何 & shared SH）
        template_gs.merge(all_instances, mode="merge")

        # 3. 为每个实例初始化 SH offset（同 shape）
        xyz_offsets = [
            torch.zeros_like(template_gs.get_xyz, requires_grad=True)
            for _ in all_instances
        ]
        scaling_offsets = [
            torch.zeros_like(template_gs.get_scaling, requires_grad=True)
            for _ in all_instances
        ]
        rotation_offsets = [
            torch.zeros_like(template_gs.get_rotation, requires_grad=True)
            for _ in all_instances
        ]
        features_dc_offsets = [
            torch.zeros_like(template_gs.get_features_dc, requires_grad=True)
            for _ in all_instances
        ]
        features_rest_offsets = [
            torch.zeros_like(template_gs.get_features_rest, requires_grad=True)
            for _ in all_instances
        ]
        opacity_offsets = [
            torch.zeros_like(template_gs.get_opacity, requires_grad=True)
            for _ in all_instances
        ]

        template_gs.set_template_id(template_id)
        template_gs.set_transforms(transforms)
        template_gs.set_xyz_offsets(xyz_offsets)
        template_gs.set_scaling_offsets(scaling_offsets)
        template_gs.set_rotation_offsets(rotation_offsets)
        template_gs.set_features_dc_offsets(features_dc_offsets)
        template_gs.set_features_rest_offsets(features_rest_offsets)
        template_gs.set_opacity_offsets(opacity_offsets)

        all_template_gs.append(template_gs)

    bg_gaussians = GaussianModel(sh_degree=3)
    bg_gaussians.load_ply(f"./data/{SCENE_NAME}/seg_inst/bg.ply")

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=np.random.randint(10000, 20000))
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 10_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--checkpoint_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000]
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    args.source_path = f"./data/{SCENE_NAME}"
    args.model_path = f"./output/{SCENE_NAME}"
    args.iterations = 10000

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        all_template_gs,
        bg_gaussians,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
