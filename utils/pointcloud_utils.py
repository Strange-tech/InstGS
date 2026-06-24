import open3d as o3d
import numpy as np


def preprocess_point_cloud(pcd, voxel_size):
    """
    下采样并估计法线，返回下采样点云与其 FPFH 特征
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    """
    用基于特征的 RANSAC 做粗配准（返回变换矩阵）
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),  # rigid
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500),
    )
    return result


def refine_registration(source, target, init_transformation, voxel_size):
    """
    使用 point-to-plane ICP 做精配准
    """
    distance_threshold = voxel_size * 0.5
    # 确保有法线（若未估计，应 estimate_normals）
    if not source.has_normals():
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )
    if not target.has_normals():
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
        )

    result_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000),
    )
    return result_icp


def fpfh_icp_registration(source_pcd, target_pcd, voxel_size=0.05, verbose=True):
    """
    整体流程：FPFH 全局配准 -> ICP 精配准
    输入：
      - source_pcd, target_pcd: open3d.geometry.PointCloud 对象
      - voxel_size: 下采样体素大小（根据点云尺度调整，单位与点云坐标一致）
    返回：
      - result_ransac: ransac 结果对象
      - result_icp: icp 结果对象
    """
    # 1. 下采样并计算特征
    src_down, src_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    tgt_down, tgt_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    if verbose:
        print("Downsampled source/target:", src_down, tgt_down)

    # 2. 全局粗配准（FPFH + RANSAC）
    result_ransac = execute_global_registration(
        src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size
    )
    if verbose:
        print("RANSAC result:")
        print(result_ransac)
        print("RANSAC transformation:\n", result_ransac.transformation)

    # 3. 使用粗配准结果作为 ICP 的初值做精配准
    result_icp = refine_registration(
        source_pcd, target_pcd, result_ransac.transformation, voxel_size
    )
    if verbose:
        print("ICP result:")
        print(result_icp)
        print("ICP transformation:\n", result_icp.transformation)
        print("Fitness:", result_icp.fitness, "Inlier RMSE:", result_icp.inlier_rmse)

    return result_ransac, result_icp
