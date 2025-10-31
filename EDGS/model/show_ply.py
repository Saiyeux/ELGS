import open3d as o3d
import numpy as np

def read_and_visualize_ply(file_path):
    # 读取PLY文件
    try:
        # 读取点云
        pcd = o3d.io.read_point_cloud(file_path)
        
        # 检查点云是否为空
        if not pcd.has_points():
            print("错误: PLY文件不包含点云数据")
            return
        
        # 打印点云信息
        print(f"点云包含 {np.asarray(pcd.points).shape[0]} 个点")
        
        # 如果点云没有颜色，添加默认颜色
        if not pcd.has_colors():
            pcd.paint_uniform_color([0, 0, 0])  # 灰色
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="PLY Point Cloud Visualization")
        
        # 添加点云到可视化
        vis.add_geometry(pcd)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.point_size = 2.0  # 设置点的大小
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # 设置背景颜色
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        print(f"读取或显示PLY文件时发生错误: {str(e)}")

if __name__ == "__main__":
    # 替换为你的PLY文件路径
    ply_file_path = "output_xyz_rgb/point_cloud.ply"
    read_and_visualize_ply(ply_file_path)