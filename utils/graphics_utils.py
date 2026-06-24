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
import math
import numpy as np
from typing import NamedTuple
from scipy.spatial.transform import Rotation as R

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

class OffsetPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    opacities : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

# def geom_transform_points_np(points, transf_matrix):
#     P, _ = points.shape
#     ones = np.ones((P, 1), dtype=points.dtype)
#     points_hom = np.concatenate([points, ones], axis=1)
#     points_out = points_hom @ transf_matrix.T
#     denom = points_out[..., 3:] + 1e-7
#     return (points_out[..., :3] / denom)


def geom_transform_quat(quats, transf_matrix):
    assert quats.shape[1] == 4 and transf_matrix.shape == (4, 4)

    # 1. 提取旋转部分
    R_np = transf_matrix[:3, :3].cpu().numpy()
    R_quat_np = R.from_matrix(R_np).as_quat()  # [x, y, z, w]
    R_quat = torch.from_numpy(R_quat_np).to(quats.device).float()

    # 2. 四元数运算函数
    def quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)
        return torch.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dim=-1,
        )

    def quat_conjugate(q):
        return torch.cat([-q[:, :3], q[:, 3:]], dim=-1)

    # 3. 进行共轭变换：q' = R * q * R⁻¹
    R_quat_batch = R_quat.unsqueeze(0).expand(quats.shape[0], 4)
    R_conj = quat_conjugate(R_quat_batch)
    q_out = quat_multiply(quat_multiply(R_quat_batch, quats), R_conj)

    # 4. 归一化结果
    q_out = q_out / q_out.norm(dim=-1, keepdim=True)
    return q_out


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))