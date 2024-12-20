import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from FFAG_Bunch import FFAG_ManageBunchAttribute


def process_step_dump_csv(folder_path, max_files=None):
    """
    处理 StepDump 数据，读取每个粒子的 .csv 文件并绘制轨迹和垂直运动图。
    :param folder_path: str, 包含 StepDump 数据的文件夹路径。
    :param max_files: int, 限制绘制的粒子文件数，默认处理所有文件。
    """
    attributes = FFAG_ManageBunchAttribute().Attribute  # 获取属性字典

    # 获取所有粒子文件（假设以 .csv 保存）
    particle_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if max_files:
        particle_files = particle_files[:max_files]  # 限制文件数

    # 创建绘图窗口
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for particle_file in particle_files:
        filepath = os.path.join(folder_path, particle_file)
        try:
            # 加载粒子数据
            data = pd.read_csv(filepath, header=None).to_numpy()

            # 按属性提取数据
            r = data[:, attributes['r']]        # 径向位置
            vr = data[:, attributes['vr']]  # 径向位置
            z = data[:, attributes['z']]        # 垂直位置
            fi = data[:, attributes['fi']]      # 方位角

            x, y = r * np.cos(fi), r * np.sin(fi)

            # 在平面轨迹图中绘制
            ax1.plot(x, y, linewidth=0.2)

            # 在垂直运动图中绘制
            ax2.plot(fi / np.pi / 2, z, linewidth=0.2)

        except Exception as e:
            print(f"Error processing file {particle_file}: {e}")
            continue

    # 设置图 1：x-y 轨迹
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Particle Trajectory (x-y)')

    # 设置图 2：fi-z 轨迹
    ax2.set_xlabel('turns')
    ax2.set_ylabel('z')
    ax2.set_title('Vertical Motion (fi-z)')

    # 显示图形
    plt.show()


def process_step_dump_bunch_csv(folder_path, max_files=5):
    """
    处理 StepDumpBunch 数据，每个时刻的 .csv 文件绘制单独的图，最多绘制 max_files 张图。
    :param folder_path: str, 包含 StepDumpBunch 数据的文件夹路径。
    :param max_files: int, 限制绘制的文件数，默认最多绘制 5 张图。
    """
    attributes = FFAG_ManageBunchAttribute().Attribute  # 获取属性字典

    # 获取所有时刻文件（假设以 .csv 保存）
    bunch_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 限制最多绘制的文件数
    if max_files:
        bunch_files = bunch_files[:max_files]

    for i, bunch_file in enumerate(bunch_files):
        filepath = os.path.join(folder_path, bunch_file)
        try:
            # 加载束团数据
            data = pd.read_csv(filepath, header=None).to_numpy()

            # 按属性提取数据
            r = data[:, attributes['r']]        # 径向位置
            vr = data[:, attributes['vr']]      # 径向速度
            z = data[:, attributes['z']]        # 垂直位置
            vz = data[:, attributes['vz']]      # 垂直速度

            # 创建绘图窗口
            fig, ax = plt.subplots(figsize=(8, 6))

            # 在平面轨迹图中绘制散点图
            scatter = ax.scatter(z, vz, s=1.0, alpha=0.8)

            # 设置图标题和轴标签
            ax.set_title(f"Bunch Trajectory for File: {bunch_file}", fontsize=12)
            ax.set_xlabel('r (radial position)')
            ax.set_ylabel('vr (radial velocity)')

            # 保存或显示图形
            plt.show()

        except Exception as e:
            print(f"Error processing file {bunch_file}: {e}")
            continue

        # 提示完成
        if i + 1 >= max_files:
            print(f"Processed the maximum number of files ({max_files}).")
            break


def process_position_dump_csv(folder_path, max_files=50):
    """
    处理 PositionDump 数据，每个穿越点的 .csv 文件绘制单独的图，最多绘制 max_files 张图。
    :param folder_path: str, 包含 PositionDump 数据的文件夹路径。
    :param max_files: int, 限制绘制的文件数，默认最多绘制 5 张图。
    """
    attributes = FFAG_ManageBunchAttribute().Attribute  # 获取属性字典

    # 获取所有穿越点文件（假设以 .csv 保存）
    crossing_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 限制最多绘制的文件数
    if max_files:
        crossing_files = crossing_files[:max_files]

    for i, crossing_file in enumerate(crossing_files):
        filepath = os.path.join(folder_path, crossing_file)
        try:
            # 加载穿越点数据
            data = pd.read_csv(filepath, header=None).to_numpy()

            # 按属性提取数据
            r = data[:, attributes['r']]        # 径向位置
            fi = data[:, attributes['fi']]      # 方位角
            z = data[:, attributes['z']]        # 垂直位置
            vr = data[:, attributes['vr']]      # 径向速度
            vz = data[:, attributes['vz']]      # 垂直速度

            # 创建绘图窗口
            plt.figure(1, figsize=(14, 6))

            # 图 1：x-y 平面轨迹
            plt.scatter(z, vz, s=1.0, alpha=0.8)

            # 显示图形
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error processing file {crossing_file}: {e}")
            continue

        # 提示完成
        if i + 1 >= max_files:
            print(f"Processed the maximum number of files ({max_files}).")
            break


def process_merged_files(folder_path, max_files=50):
    """
    通用函数，根据 merged_files 文件夹中的文件名前缀调用相应的处理函数。
    :param folder_path: str, 包含 merged_files 的文件夹路径。
    :param max_files: int, 限制绘制的文件数，默认最多处理 50 个文件。
    """
    merged_files_path = os.path.join(folder_path, "merged_files")
    if not os.path.exists(merged_files_path):
        print(f"Error: The folder '{merged_files_path}' does not exist.")
        return

    # 获取所有 .csv 文件
    csv_files = [f for f in os.listdir(merged_files_path) if f.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in the folder '{merged_files_path}'.")
        return

    # 分类文件名
    step_dump_files = [f for f in csv_files if f.startswith('particle')]
    step_bunch_files = [f for f in csv_files if f.startswith('step')]
    position_dump_files = [f for f in csv_files if f.startswith('crossing_angle')]

    # 处理 StepDump 数据
    if step_dump_files:
        print("Processing StepDump files...")
        process_step_dump_csv(merged_files_path, max_files)

    # 处理 StepDumpBunch 数据
    if step_bunch_files:
        print("Processing StepDumpBunch files...")
        process_step_dump_bunch_csv(merged_files_path, max_files)

    # 处理 PositionDump 数据
    if position_dump_files:
        print("Processing PositionDump files...")
        process_position_dump_csv(merged_files_path, max_files)

    print("Processing complete.")


# 调用示例
process_merged_files("./step_dump_sc_2", max_files=50)
