#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from einops import repeat
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from inst_gaussian_rasterization import InstGaussianRasterizationSettings, InstGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.inst_gaussian_model import InstGaussianModel
from utils.sh_utils import eval_sh

class GaussianParams:
    def __init__(self, means3D, means2D, scales, rotations, shs, opacities, colors_precomp=None, cov3D_precomp=None):
        self.means3D = means3D
        self.means2D = means2D
        self.scales = scales
        self.rotations = rotations
        self.shs = shs
        self.opacities = opacities
        self.colors_precomp = torch.Tensor([]) if colors_precomp is None else colors_precomp
        self.cov3D_precomp = torch.Tensor([]) if cov3D_precomp is None else cov3D_precomp
    
    def to(self, device):
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))


class templateGaussianParams:
    def __init__(self, means3D, means2D, scales, rotations, shs, opacities, transforms, xyz_offsets, sh_offsets, opacity_offsets, colors_precomp=None, cov3D_precomp=None):
        self.means3D = means3D
        self.means2D = means2D
        self.scales = scales
        self.rotations = rotations
        self.shs = shs
        self.opacities = opacities
        self.transforms = transforms
        self.xyz_offsets = xyz_offsets
        self.sh_offsets = sh_offsets
        self.opacity_offsets = opacity_offsets
        self.colors_precomp = torch.Tensor([]) if colors_precomp is None else colors_precomp
        self.cov3D_precomp = torch.Tensor([]) if cov3D_precomp is None else cov3D_precomp

    def to(self, device):
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))

    
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii
        }
    
    return out

def instanced_render(
    viewpoint_camera,
    all_temp_gs: list,
    bg_gs: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    filtered_mask=None,
):
    """
    Instanced Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    all_inst_xyz = torch.cat(
        [temp_gs.get_full_xyz for temp_gs in all_temp_gs], dim=0
    ).to("cuda")
    all_inst_scaling = torch.cat(
        [temp_gs.get_full_scaling for temp_gs in all_temp_gs], dim=0
    ).to("cuda")
    all_inst_rotation = torch.cat(
        [temp_gs.get_full_rotation for temp_gs in all_temp_gs], dim=0
    ).to("cuda")
    all_inst_opacity = torch.cat(
        [temp_gs.get_full_opacity for temp_gs in all_temp_gs], dim=0
    ).to("cuda")
    all_inst_features = torch.cat(
        [temp_gs.get_full_features for temp_gs in all_temp_gs], dim=0
    ).to("cuda")

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if bg_gs is not None:
        bg_screenspace_points = (
            torch.zeros_like(
                bg_gs.get_xyz,
                dtype=bg_gs.get_xyz.dtype,
                requires_grad=False,
                device="cuda",
            )
            + 0
        )

    screenspace_points = (
        torch.zeros_like(
            all_inst_xyz, dtype=all_inst_xyz.dtype, requires_grad=True, device="cuda"
        )
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=all_temp_gs[0].active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if bg_gs is not None:
        # If background Gaussians are provided, append them to the instance Gaussians.
        means3D = torch.cat([bg_gs.get_xyz, all_inst_xyz], dim=0).to("cuda")
        means2D = torch.cat([bg_screenspace_points, screenspace_points], dim=0).to(
            "cuda"
        )
        opacity = torch.cat([bg_gs.get_opacity, all_inst_opacity], dim=0).to("cuda")
    else:
        means3D = all_inst_xyz
        means2D = screenspace_points
        opacity = all_inst_opacity

    if filtered_mask is not None:
        new_opacity = opacity.detach().clone()
        new_opacity[filtered_mask, :] = 0
        opacity = new_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = all_temp_gs[0].covariance_activation(
            all_inst_scaling, scaling_modifier, all_inst_rotation
        )
    else:
        if bg_gs is not None:
            scales = torch.cat([bg_gs.get_scaling, all_inst_scaling], dim=0).to("cuda")
            rotations = torch.cat([bg_gs.get_rotation, all_inst_rotation], dim=0).to(
                "cuda"
            )
        else:
            scales = all_inst_scaling
            rotations = all_inst_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = all_inst_features.transpose(1, 2).view(
                -1, 3, (all_temp_gs[0].max_sh_degree + 1) ** 2
            )
            dir_pp = all_inst_xyz - viewpoint_camera.camera_center.repeat(
                all_inst_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(
                all_temp_gs[0].active_sh_degree, shs_view, dir_pp_normalized
            )
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if bg_gs is not None:
                shs = torch.cat([bg_gs.get_features, all_inst_features], dim=0).to(
                    "cuda"
                )
            else:
                shs = all_inst_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    # clip the shared gaussian
    l = all_inst_xyz.shape[0]
    radii = radii[-l:]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


def instanced_render_cuda(
    viewpoint_camera,
    temp_gs: InstGaussianModel,
    bg_gs: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    filtered_mask=None,
):
    """
    Instanced Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    screenspace_points = torch.zeros_like(temp_gs.get_xyz, dtype=temp_gs.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    raster_settings = InstGaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=temp_gs.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = InstGaussianRasterizer(raster_settings=raster_settings)

    means3D = temp_gs.get_xyz
    means2D = screenspace_points
    opacity = temp_gs._opacity # 可不敢改！！！！
        

    if filtered_mask is not None:
        new_opacity = opacity.detach().clone()
        new_opacity[filtered_mask, :] = 0
        opacity = new_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    
    scales = temp_gs.get_scaling
    rotations = temp_gs.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = temp_gs.get_features
    else:
        colors_precomp = override_color

    transforms = temp_gs.get_transforms.reshape(-1, 16)

    # print(temp_gs.get_features_dc_offsets.shape)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    t_g_p = templateGaussianParams(
        temp_gs.get_xyz, None, temp_gs.get_scaling, temp_gs.get_rotation,
        temp_gs.get_features, temp_gs._opacity, transforms,
        temp_gs.get_xyz_offsets, temp_gs.get_features_dc_offsets, temp_gs.get_opacity_offsets)
    
    if bg_gs is not None:
        g_p = GaussianParams(
            bg_gs.get_xyz, None, bg_gs.get_scaling, bg_gs.get_rotation, 
            bg_gs.get_features, bg_gs._opacity)
    else:
        g_p = GaussianParams(
            torch.empty(0, device="cuda"),
            None,
            torch.empty(0, device="cuda"),
            torch.empty(0, device="cuda"),
            torch.empty(0, device="cuda"),
            torch.empty(0, device="cuda"),
        )
        
    rendered_image, radii, depth_image = rasterizer(t_g_p, g_p)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": None,
        # "visibility_filter": radii > 0,
        # "radii": radii,
    }

