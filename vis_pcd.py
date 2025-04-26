# viz_pcd.py
import numpy as np
import open3d as o3d
import pathlib

def load_pointcloud(folder: pathlib.Path):
    xyz = np.load(folder / 'xyz.npy')  # (N,3)
    rgb = np.load(folder / 'rgb.npy')  # (N,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(xyz)
    pcd.colors  = o3d.utility.Vector3dVector(rgb)
    return pcd

if __name__ == "__main__":
    folder = pathlib.Path('/tmp/pcd_dump')   # 저장했던 경로
    pcd = load_pointcloud(folder)

    print(pcd)        # PointCloud with N points
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Dreamer PointCloud",
        width=800,
        height=600,
        point_show_normal=False,
    )