from utils.pointcloud_utils import fpfh_icp_registration
import open3d as o3d
import numpy as np
import json

SCENE_NAME = "figurines"

if __name__ == "__main__":

    scene_graph = []

    template_group = {}
    template_group["template_id"] = "1"
    template_group["instances"] = []
    template_group["instances"].append({"instance_id": "1", "transform": np.eye(4).tolist()})

    for i in range(2, 18):

        # 读入点云（支持 ply, pcd 等）
        src = o3d.io.read_point_cloud(f"./data/{SCENE_NAME}/seg_inst/{i}.ply")
        src.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1, 0.706, 0]]), (len(src.points), 1)))
        tgt = o3d.io.read_point_cloud(f"./data/{SCENE_NAME}/seg_inst/1.ply")
        tgt.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 0.651, 0.929]]), (len(tgt.points), 1)))

        # 根据你的点云尺度选择 voxel_size（大场景用 0.1~0.5，小物体用 0.01~0.05）
        voxel_size = 0.05

        ransac_res, icp_res = fpfh_icp_registration(
            src, tgt, voxel_size=voxel_size, verbose=True
        )

        # 应用最终变换并保存结果
        src_transformed = src.transform(icp_res.transformation)
        # o3d.io.write_point_cloud("source_aligned.ply", src_transformed)


        o3d.visualization.draw_geometries(
            [src_transformed, tgt], window_name="FPFH + ICP Alignment"
        )

        template_group["instances"].append({"instance_id": str(i), "transform": icp_res.transformation.tolist()})

    scene_graph.append(template_group)

    # save the scene graph to a file
    scene_graph_path = f"./data/{SCENE_NAME}/scene_graph.json"
    with open(scene_graph_path, "w") as f:
        json.dump(scene_graph, f, indent=4)
    print(f"Scene graph saved to {scene_graph_path}")
