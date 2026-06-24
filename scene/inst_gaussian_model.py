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
import open3d as o3d
import torch
import numpy as np
from utils.general_utils import (
    inverse_sigmoid,
    get_expon_lr_func,
    build_rotation,
    random_point_sampling,
)
from torch import nn
import os
import re
from copy import deepcopy
from utils.system_utils import mkdir_p
from utils.graphics_utils import (
    geom_transform_points,
)
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, OffsetPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.gaussian_model import GaussianModel
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion


def compress(f_rest_offsets):

    compressed_f_rest_offsets = []

    for f_r_o in f_rest_offsets:
        flattened = f_r_o.view(f_r_o.shape[0], -1)
        mask = ~(flattened == 0).all(dim=1)
        indices = mask.nonzero(as_tuple=True)[0].to(torch.int32)  # int32, 4 bytes
        values = f_r_o[mask]  # float32, 4 bytes
        compressed_f_rest_offsets.append((indices, values))

    return compressed_f_rest_offsets


class InstGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            # symm = strip_symmetric(actual_covariance)
            return actual_covariance

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)  # points to be optimized (bg + templates)
        self._mask = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.template_id = None
        # transforms: [trans1, trans2, ...]
        self.transforms = []
        self._xyz_offsets = []
        self._scaling_offsets = []
        self._rotation_offsets = []
        self._features_dc_offsets = []
        self._features_rest_offsets = []
        self._opacity_offsets = []
        self.setup_functions()

    def capture(self):
        return (
            self.template_id,
            self.active_sh_degree,
            self._xyz,
            self._mask,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.transforms,
            self._xyz_offsets,
            self._scaling_offsets,
            self._rotation_offsets,
            self._features_dc_offsets,
            self._features_rest_offsets,
            self._opacity_offsets,
        )

    # def min_capture(self):
    #     return (
    #         self.template_id,
    #         self.active_sh_degree,
    #         self._xyz,
    #         self._scaling,
    #         self._rotation,
    #         self.transforms,
    #         self._features_dc_offsets,
    #         compress(self._features_rest_offsets),
    #         self._opacity_offsets,
    #     )

    def restore(self, model_args, training_args):
        (
            self.template_id,
            self.active_sh_degree,
            self._xyz,
            self._mask,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            opt_dict,
            self.spatial_lr_scale,
            self.transforms,
            self._xyz_offsets,
            self._scaling_offsets,
            self._rotation_offsets,
            self._features_dc_offsets,
            self._features_rest_offsets,
            self._opacity_offsets,
        ) = model_args
        if training_args:
            self.training_setup(training_args)
        self.instances_num = len(self.transforms)
        # self.optimizer.load_state_dict(opt_dict)

    def set_template_id(self, template_id):
        self.template_id = template_id

    def set_transforms(self, transforms):
        self.transforms = transforms
        self.instances_num = len(transforms)

    def set_xyz_offsets(self, xyz_offsets):
        for offset in xyz_offsets:
            self._xyz_offsets.append(nn.Parameter(offset.requires_grad_(True)))

    def set_scaling_offsets(self, scaling_offsets):
        for offset in scaling_offsets:
            self._scaling_offsets.append(nn.Parameter(offset.requires_grad_(True)))

    def set_rotation_offsets(self, rotation_offsets):
        for offset in rotation_offsets:
            self._rotation_offsets.append(nn.Parameter(offset.requires_grad_(True)))

    def set_features_dc_offsets(self, features_dc_offsets):
        for offset in features_dc_offsets:
            self._features_dc_offsets.append(nn.Parameter(offset.requires_grad_(True)))

    def set_features_rest_offsets(self, features_rest_offsets):
        for offset in features_rest_offsets:
            self._features_rest_offsets.append(
                nn.Parameter(offset.requires_grad_(True))
            )

    def set_opacity_offsets(self, opacity_offsets):
        for offset in opacity_offsets:
            self._opacity_offsets.append(nn.Parameter(offset.requires_grad_(True)))

    def get_offset_point_clouds(self):
        offset_point_clouds = []
        for idx in range(self.instances_num):
            o_pc = OffsetPointCloud(self._xyz_offsets[idx].detach().cpu().numpy(), self._features_dc_offsets[idx].detach().cpu().numpy(), self._opacity_offsets[idx].detach().cpu().numpy())
            offset_point_clouds.append(o_pc)
        return offset_point_clouds

    # CPU instancing
    def instancing(self):
        all_instances_xyz = []
        all_instances_scaling = []
        all_instances_rotation = []
        all_instances_opacity = []
        all_instances_features_dc = []
        all_instances_features_rest = []

        for idx, tran in enumerate(self.transforms):
            # xyz
            trans_xyz = geom_transform_points(self._xyz, tran) + self._xyz_offsets[idx] if self._xyz_offsets else geom_transform_points(self._xyz, tran)
            all_instances_xyz.append(trans_xyz)
            # rotation and scaling
            trans_scaling = self._scaling + self._scaling_offsets[idx] if self._scaling_offsets else self._scaling
            all_instances_scaling.append(trans_scaling)

            R_local = quaternion_to_matrix(self.get_rotation)
            R_new = tran.T[:3, :3] @ R_local
            quat_new = matrix_to_quaternion(R_new) + self._rotation_offsets[idx] if self._rotation_offsets else matrix_to_quaternion(R_new)
            all_instances_rotation.append(quat_new)
            # opacity stays same
            # trans_opacity_offsets = self._opacity
            trans_opacity_offsets = (
                self._opacity + self._opacity_offsets[idx]
            ) if self._opacity_offsets else self._opacity
            all_instances_opacity.append(trans_opacity_offsets)
            # feature_dc
            trans_features_dc = (
                self._features_dc + self._features_dc_offsets[idx]
            ) if self._features_dc_offsets else self._features_dc
            all_instances_features_dc.append(trans_features_dc)
            # feature_rest
            trans_features_rest = (
                self._features_rest + self._features_rest_offsets[idx]
            ) if self._features_rest_offsets else self._features_rest
            all_instances_features_rest.append(trans_features_rest)

        self.full_xyz = torch.cat(all_instances_xyz, dim=0)
        self.full_scaling = torch.cat(all_instances_scaling, dim=0)
        self.full_rotation = torch.cat(all_instances_rotation, dim=0)
        self.full_opacity = torch.cat(all_instances_opacity, dim=0)
        self.full_features_dc = torch.cat(all_instances_features_dc, dim=0)
        self.full_features_rest = torch.cat(all_instances_features_rest, dim=0)

        self.max_radii2D = torch.zeros((self.full_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros(
            (self.full_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.full_xyz.shape[0], 1), device="cuda")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_full_scaling(self):
        return self.scaling_activation(self.full_scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_full_rotation(self):
        return self.rotation_activation(self.full_rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_full_xyz(self):
        return self.full_xyz

    @property
    def get_mask(self):
        return self._mask

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_full_features(self):
        full_features_dc = self.full_features_dc
        full_features_rest = self.full_features_rest
        return torch.cat((full_features_dc, full_features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_full_opacity(self):
        return self.opacity_activation(self.full_opacity)
    
    @property
    def get_transforms(self):
        return torch.stack(self.transforms)

    @property
    def get_xyz_offsets(self):
        return torch.stack(self._xyz_offsets) if self._xyz_offsets else torch.tensor([])
    
    @property
    def get_scaling_offsets(self):
        return torch.stack(self._scaling_offsets) if self._scaling_offsets else torch.tensor([])

    @property
    def get_rotation_offsets(self):
        return torch.stack(self._rotation_offsets) if self._rotation_offsets else torch.tensor([])

    @property
    def get_features_dc_offsets(self):
        return torch.stack(self._features_dc_offsets) if self._features_dc_offsets else torch.tensor([])

    @property
    def get_opacity_offsets(self):
        return torch.stack(self._opacity_offsets) if self._opacity_offsets else torch.tensor([])

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_offsets(self, pcd: OffsetPointCloud):
        self.spatial_lr_scale = 0.1
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color.squeeze()
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        mask = torch.ones(
            (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
        )

        opacities = torch.tensor(np.asarray(pcd.opacities)).float().cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        mask = torch.ones(
            (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
        )

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        for idx, offset in enumerate(self._xyz_offsets):
            l.append(
                {
                    "params": [offset],
                    "lr": training_args.position_lr_init * self.spatial_lr_scale,
                    "name": f"xyz_offset_{idx}",
                }
            )

        for idx, offset in enumerate(self._scaling_offsets):
            l.append(
                {
                    "params": [offset],
                    "lr": training_args.scaling_lr,
                    "name": f"scaling_offset_{idx}",
                }
            )

        for idx, offset in enumerate(self._rotation_offsets):
            l.append(
                {
                    "params": [offset],
                    "lr": training_args.rotation_lr,
                    "name": f"rotation_offset_{idx}",
                }
            )

        for idx, offset in enumerate(self._features_dc_offsets):
            l.append(
                {
                    "params": [offset],
                    "lr": training_args.feature_lr,
                    "name": f"f_dc_offset_{idx}",
                }
            )

        for idx, offset in enumerate(self._features_rest_offsets):
            l.append(
                {
                    "params": [offset],
                    "lr": training_args.feature_lr / 20,
                    "name": f"f_rest_offset_{idx}",
                }
            )

        for idx, offset in enumerate(self._opacity_offsets):
            l.append(
                {
                    "params": [offset],
                    "lr": training_args.opacity_lr,
                    "name": f"opacity_offset_{idx}",
                }
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path, instancing=False):
        mkdir_p(os.path.dirname(path))

        if instancing:
            xyz = self.full_xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = (
                self.full_features_dc.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            f_rest = (
                self.full_features_rest.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            opacities = self.full_opacity.detach().cpu().numpy()
            scale = self.full_scaling.detach().cpu().numpy()
            rotation = self.full_rotation.detach().cpu().numpy()
        else:
            xyz = self._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = (
                self._features_dc.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            f_rest = (
                self._features_rest.detach()
                .transpose(1, 2)
                .flatten(start_dim=1)
                .contiguous()
                .cpu()
                .numpy()
            )
            opacities = self._opacity.detach().cpu().numpy()
            scale = self._scaling.detach().cpu().numpy()
            rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, mask, normals, f_dc, f_rest, opacities, scale, rotation), axis=1) if has_mask else np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def save_mask(self, path):
        mkdir_p(os.path.dirname(path))
        mask = self._mask.detach().cpu().numpy()
        np.save(path, mask)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )

        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        for idx, offset in enumerate(self._features_dc_offsets):
            self._features_dc_offsets[idx] = optimizable_tensors[f"f_dc_offset_{idx}"]
        for idx, offset in enumerate(self._features_rest_offsets):
            self._features_rest_offsets[idx] = optimizable_tensors[
                f"f_rest_offset_{idx}"
            ]
        for idx, offset in enumerate(self._opacity_offsets):
            self._opacity_offsets[idx] = optimizable_tensors[f"opacity_offset_{idx}"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[
            valid_points_mask.repeat(self.instances_num)
        ]
        self.denom = self.denom[valid_points_mask.repeat(self.instances_num)]
        self.max_radii2D = self.max_radii2D[
            valid_points_mask.repeat(self.instances_num)
        ]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_features_dc_offsets,
        new_features_rest_offsets,
        new_opacity_offsets,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        for idx, offset in enumerate(new_features_dc_offsets):
            d[f"f_dc_offset_{idx}"] = offset
        for idx, offset in enumerate(new_features_rest_offsets):
            d[f"f_rest_offset_{idx}"] = offset
        for idx, offset in enumerate(new_opacity_offsets):
            d[f"opacity_offset_{idx}"] = offset

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        for idx, offset in enumerate(self._features_dc_offsets):
            self._features_dc_offsets[idx] = optimizable_tensors[f"f_dc_offset_{idx}"]
        for idx, offset in enumerate(self._features_rest_offsets):
            self._features_rest_offsets[idx] = optimizable_tensors[
                f"f_rest_offset_{idx}"
            ]
        for idx, offset in enumerate(self._opacity_offsets):
            self._opacity_offsets[idx] = optimizable_tensors[f"opacity_offset_{idx}"]

        self.xyz_gradient_accum = torch.zeros(
            (self._xyz.shape[0] * self.instances_num, 1), device="cuda"
        )
        self.denom = torch.zeros(
            (self._xyz.shape[0] * self.instances_num, 1), device="cuda"
        )
        self.max_radii2D = torch.zeros(
            (self._xyz.shape[0] * self.instances_num), device="cuda"
        )

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_features_dc_offsets = deepcopy(self._features_dc_offsets)
        new_features_rest_offsets = deepcopy(self._features_rest_offsets)
        new_opacity_offsets = deepcopy(self._opacity_offsets)

        for idx, offset in enumerate(self._features_dc_offsets):
            new_features_dc_offsets[idx] = offset[selected_pts_mask].repeat(N, 1, 1)
        for idx, offset in enumerate(self._features_rest_offsets):
            new_features_rest_offsets[idx] = offset[selected_pts_mask].repeat(N, 1, 1)
        for idx, offset in enumerate(self._opacity_offsets):
            new_opacity_offsets[idx] = offset[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_features_dc_offsets,
            new_features_rest_offsets,
            new_opacity_offsets,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_features_dc_offsets = deepcopy(self._features_dc_offsets)
        new_features_rest_offsets = deepcopy(self._features_rest_offsets)
        new_opacity_offsets = deepcopy(self._opacity_offsets)

        for idx, offset in enumerate(self._features_dc_offsets):
            new_features_dc_offsets[idx] = offset[selected_pts_mask]
        for idx, offset in enumerate(self._features_rest_offsets):
            new_features_rest_offsets[idx] = offset[selected_pts_mask]
        for idx, offset in enumerate(self._opacity_offsets):
            new_opacity_offsets[idx] = offset[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_features_dc_offsets,
            new_features_rest_offsets,
            new_opacity_offsets,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # merge grads to the shape like _xyz
        assert self._xyz.shape[0] * self.instances_num == grads.shape[0]

        all_inst_grads = torch.zeros(
            (self.instances_num, self._xyz.shape[0]), device="cuda"
        )
        all_inst_max_radii2D = torch.zeros(
            (self.instances_num, self._xyz.shape[0]), device="cuda"
        )
        for i in range(self.instances_num):
            start_idx = i * self._xyz.shape[0]
            end_idx = start_idx + self._xyz.shape[0]
            inst_grads = grads[start_idx:end_idx].squeeze(-1)
            inst_max_radii2D = self.max_radii2D[start_idx:end_idx]
            all_inst_grads[i] = inst_grads
            all_inst_max_radii2D[i] = inst_max_radii2D
        grads = all_inst_grads.mean(dim=0)
        max_radii2D = all_inst_max_radii2D.max(dim=0).values

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor_grad, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor_grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def merge(self, all_instances, mode="merge"):
        # mode: "max", "merge"
        if mode == "max":
            num = 0
            max_idx = -1
            for idx, inst in enumerate(all_instances):
                xyz = inst.get_xyz  # shape: [M, 3]
                color = inst.get_features_dc
                if xyz.shape[0] > num:
                    num = xyz.shape[0]
                    max_idx = idx
            max_xyz = all_instances[max_idx].get_xyz
            self.densify_max = max_xyz.shape[0]
            # 先搞他个一半再说
            # sampling_idx = random_point_sampling(max_xyz, num // 2)
            # sampled_xyz = max_xyz[sampling_idx]
            max_color = all_instances[max_idx].get_features_dc.squeeze()
            # sampled_color = max_color[sampling_idx]
            pcd = BasicPointCloud(
                points=max_xyz.detach().cpu().numpy(),
                colors=max_color.detach().cpu().numpy(),
                normals=np.zeros((num, 3)),
            )
            self.create_from_pcd(pcd, spatial_lr_scale=0.1)
        elif mode == "merge":
            all_xyz_list = []
            all_color_list = []
            num_pts = 0
            for inst in all_instances:
                all_xyz_list.append(inst.get_xyz)
                all_color_list.append(inst.get_features_dc)
                num_pts += inst.get_xyz.shape[0]
            all_xyz = torch.cat(all_xyz_list, dim=0)  # shape: [total_M, 3]
            all_color = torch.cat(all_color_list, dim=0)

            print("FPS...")
            num_pts = int(num_pts / len(all_instances)) # average num_pts
            sampling_idx = random_point_sampling(all_xyz, num_pts)
            sampled_xyz = all_xyz[sampling_idx]
            sampled_color = all_color[sampling_idx].squeeze()
            pcd = BasicPointCloud(
                points=sampled_xyz.detach().cpu().numpy(),
                colors=sampled_color.detach().cpu().numpy(),
                normals=np.zeros((num_pts, 3)),
            )
            self.create_from_pcd(pcd, spatial_lr_scale=0.1)
            print("Sampling Done.")
        elif mode == "all":
            all_xyz_list = []
            all_color_list = []
            num_pts = 0
            for inst in all_instances:
                all_xyz_list.append(inst.get_xyz)
                all_color_list.append(inst.get_features_dc)
                num_pts += inst.get_xyz.shape[0]
            all_xyz = torch.cat(all_xyz_list, dim=0)  # shape: [total_M, 3]
            all_color = torch.cat(all_color_list, dim=0)
            pcd = BasicPointCloud(
                points=all_xyz.detach().cpu().numpy(),
                colors=all_color.squeeze().detach().cpu().numpy(),
                normals=np.zeros((num_pts, 3)),
            )
            self.create_from_pcd(pcd, spatial_lr_scale=0.1)

    def offset_loss(self):
        offset_loss = 0.0
        # for offset in self._xyz_offsets:
        #     offset_loss += torch.mean(torch.abs(offset))
        # for offset in self._scaling_offsets:
        #     offset_loss += torch.mean(torch.abs(offset))
        for offset in self._rotation_offsets:
            offset_loss += torch.mean(torch.abs(offset))
        for offset in self._features_dc_offsets:
            offset_loss += torch.mean(torch.abs(offset))
        for offset in self._features_rest_offsets:
            offset_loss += torch.mean(torch.abs(offset))
        # for offset in self._opacity_offsets:
        #     offset_loss += torch.mean(torch.abs(offset))
        return offset_loss
