import os
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from numpy.polynomial import Polynomial
from FFAG_HighOrder import calculate_s, Generate_Bmaps, Generate_Bmap_Bz, BmapParams_to_ConvexParams, Generate_Bmaps_MultiProcess


# 设置命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Process Bmap coefficients and generate plots.")
    parser.add_argument('-j', type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument('-f', type=str, required=True, help="Folder to save the coefficient matrices.")
    parser.add_argument('-m', type=int, required=True, help="Maximum order for the coefficients.")
    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 获取 JSON 文件所在目录
    json_dir = os.path.dirname(os.path.abspath(args.j))

    # 如果 -f 是相对路径，计算绝对路径
    if not os.path.isabs(args.f):
        output_dir = os.path.join(json_dir, args.f)  # 相对路径拼接到 JSON 文件目录
    else:
        output_dir = args.f  # 已经是绝对路径，直接使用

    # 确保保存文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取配置文件
    with open(args.j, 'r') as f:
        config_data = json.load(f)

    Bmap = config_data['Bmap']
    rmin_max_step_m = Bmap["rmin_max_step_m"]
    SpiralAngle_deg = Bmap["SpiralAngle_deg"]
    theta_step_rad = Bmap["theta_step_rad"]
    rmin, rmax, rstep = rmin_max_step_m[0], rmin_max_step_m[1], rmin_max_step_m[2]

    convex_params = BmapParams_to_ConvexParams(Bmap)

    # 计算r轴步数
    n_r_steps = int((rmax - rmin) / rstep) + 1  # 确保包括rmax
    n_fi_steps = int((np.pi * 2 / Bmap["NSector"] - 0.0) / theta_step_rad) + 1  # 确保包括fi_max

    fi_axis = np.linspace(0, np.pi * 2 / Bmap["NSector"], n_fi_steps)
    r_axis = np.linspace(rmin, rmax, n_r_steps)

    s_values = calculate_s(r_axis, 0.0, SpiralAngle_deg)
    s_values_FitCoef = Polynomial.fit(r_axis, s_values, 5)

    # 调用函数并打印结果
    Br_Taylor_matrices, Bz_Taylor_matrices, Bfi_Taylor_matrices = Generate_Bmaps_MultiProcess(args.m, r_axis, fi_axis,
                                                                                 convex_params, s_values_FitCoef,
                                                                                 Bmap['coefficients_tupel'])
    SaveData, _, _ = Generate_Bmap_Bz(r_axis, fi_axis, convex_params, s_values_FitCoef, Bmap['coefficients_tupel'])

    # 按每一阶分别保存 Br, Bz, Bfi 的系数矩阵为单独的 .npz 文件
    for order in range(len(Br_Taylor_matrices)):
        np.savez(os.path.join(output_dir, f"Br_order_{order}.npz"), Br_Taylor_coeff=Br_Taylor_matrices[order])
        np.savez(os.path.join(output_dir, f"Bz_order_{order}.npz"), Bz_Taylor_coeff=Bz_Taylor_matrices[order])
        np.savez(os.path.join(output_dir, f"Bfi_order_{order}.npz"),
                 Bfi_Taylor_coeff=Bfi_Taylor_matrices[order])

    print(f"系数矩阵按阶数分别保存为 npz 文件到文件夹: {output_dir}")

    # 定义表头信息
    header = f'unitR=m unitFi=rad unitB=T Nsectors={Bmap["NSector"]}'

    # 保存 Bmap.txt 文件，并添加表头
    np.savetxt(os.path.join(output_dir, "Bmap.txt"), SaveData, header=header, comments='')
    print(f"中平面 Bz 保存为 txt 文件到文件夹: {output_dir}")

    # 将极坐标转换为笛卡尔坐标
    mesh_f, mesh_r = np.meshgrid(fi_axis, r_axis)
    mesh_x = mesh_r * np.cos(mesh_f)
    mesh_y = mesh_r * np.sin(mesh_f)

    # 创建三维图
    fig = plt.figure(figsize=(18, 6))

    # 绘制 Bz_Taylor_matrices[0] 的曲面图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(mesh_x, mesh_y, Bz_Taylor_matrices[0], cmap='viridis', edgecolor='none')
    ax1.set_title('Bz Taylor Coefficients (Order 0)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_zlabel('Bz')

    # 绘制 Br_Taylor_matrices[0] 的曲面图
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(mesh_x, mesh_y, Br_Taylor_matrices[0], cmap='plasma', edgecolor='none')
    ax2.set_title('Br Taylor Coefficients (Order 0)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_zlabel('Br')

    # 绘制 Bfi_Taylor_matrices[0] 的曲面图
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(mesh_x, mesh_y, Bfi_Taylor_matrices[0], cmap='inferno', edgecolor='none')
    ax3.set_title('Bfi Taylor Coefficients (Order 0)')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_zlabel('Bfi')

    # 显示图像
    plt.tight_layout()
    plt.show()
# def main():
#     # 解析命令行参数
#     args = parse_args()
#
#     # 读取配置文件
#     with open(args.j, 'r') as f:
#         config_data = json.load(f)
#
#     if not os.path.exists(args.f):
#         os.makedirs(args.f)
#
#     Bmap = config_data['Bmap']
#
#     # 示例
#     convex_params = BmapParams_to_ConvexParams(Bmap)
#
#     fi_axis = np.linspace(0, np.pi * 2 / Bmap["NSector"], 3000)
#     r_axis = np.linspace(6.5, 9.0, 200)
#
#     s_values = calculate_s(r_axis, 0.0, 40.0)
#     s_values_FitCoef = Polynomial.fit(r_axis, s_values, 5)
#
#     # 调用函数并打印结果
#     Br_Taylor_matrices, Bz_Taylor_matrices, Bfi_Taylor_matrices = Generate_Bmaps(args.m, r_axis, fi_axis,
#                                                                                  convex_params, s_values_FitCoef,
#                                                                                  Bmap['coefficients_tupel'])
#     SaveData, _, _ = Generate_Bmap_Bz(r_axis, fi_axis, convex_params, s_values_FitCoef, Bmap['coefficients_tupel'])
#
#     # 按每一阶分别保存 Br, Bz, Bfi 的系数矩阵为单独的 .npz 文件
#     for order in range(len(Br_Taylor_matrices)):
#         np.savez(os.path.join(args.f, f"Br_order_{order}.npz"), Br_Taylor_coeff=Br_Taylor_matrices[order])
#         np.savez(os.path.join(args.f, f"Bz_order_{order}.npz"), Bz_Taylor_coeff=Bz_Taylor_matrices[order])
#         np.savez(os.path.join(args.f, f"Bfi_order_{order}.npz"),
#                  Bfi_Taylor_coeff=Bfi_Taylor_matrices[order])
#
#     print(f"系数矩阵按阶数分别保存为 npz 文件到文件夹: {args.f}")
#
#     # 定义表头信息
#     header = f'unitR=m unitFi=rad unitB=T Nsectors={Bmap["NSector"]}'
#
#     # 保存 Bmap.txt 文件，并添加表头
#     np.savetxt(os.path.join(args.f, "Bmap.txt"), SaveData, header=header, comments='')
#     print(f"中平面 Bz 保存为 txt 文件到文件夹: {args.f}")
#
#     # 将极坐标转换为笛卡尔坐标
#     mesh_f, mesh_r = np.meshgrid(fi_axis, r_axis)
#     mesh_x = mesh_r * np.cos(mesh_f)
#     mesh_y = mesh_r * np.sin(mesh_f)
#
#     # 创建三维图
#     fig = plt.figure(figsize=(18, 6))
#
#     # 绘制 Bz_Taylor_matrices[0] 的曲面图
#     ax1 = fig.add_subplot(131, projection='3d')
#     ax1.plot_surface(mesh_x, mesh_y, Bz_Taylor_matrices[0], cmap='viridis', edgecolor='none')
#     ax1.set_title('Bz Taylor Coefficients (Order 0)')
#     ax1.set_xlabel('x (m)')
#     ax1.set_ylabel('y (m)')
#     ax1.set_zlabel('Bz')
#
#     # 绘制 Br_Taylor_matrices[0] 的曲面图
#     ax2 = fig.add_subplot(132, projection='3d')
#     ax2.plot_surface(mesh_x, mesh_y, Br_Taylor_matrices[0], cmap='plasma', edgecolor='none')
#     ax2.set_title('Br Taylor Coefficients (Order 0)')
#     ax2.set_xlabel('x (m)')
#     ax2.set_ylabel('y (m)')
#     ax2.set_zlabel('Br')
#
#     # 绘制 Bfi_Taylor_matrices[0] 的曲面图
#     ax3 = fig.add_subplot(133, projection='3d')
#     ax3.plot_surface(mesh_x, mesh_y, Bfi_Taylor_matrices[0], cmap='inferno', edgecolor='none')
#     ax3.set_title('Bfi Taylor Coefficients (Order 0)')
#     ax3.set_xlabel('x (m)')
#     ax3.set_ylabel('y (m)')
#     ax3.set_zlabel('Bfi')
#
#     # 显示图像
#     plt.tight_layout()
#     plt.show()


if __name__ == "__main__":
    main()
