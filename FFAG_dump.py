import os
import sys
import time
import numpy as np
from mpi4py import MPI
import argparse
from FFAG_Bunch import FFAG_ManageBunchAttribute


class StepDump:
    def __init__(self, step_interval, num_particles_to_dump_global=1000, save_folder="dump_data"):
        """
        StepDump 类，用于在 RK 积分过程中每隔一定的 step 进行一次粒子 Dump 操作。
        :param step_interval: int, 每隔多少步进行一次 Dump。
        :param num_particles_to_dump: int, 每次 Dump 保存前 n 个粒子的状态。
        :param save_folder: str, 保存文件的文件夹路径，不包含 rank，rank 号会自动添加为子文件夹。
        """
        self.step_interval = step_interval

        self.save_folder = save_folder

        # 获取 MPI 信息
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        #
        # 计算每个线程需要保存的粒子数量
        self.num_particles_to_dump = num_particles_to_dump_global // size
        # 创建保存路径，如果文件夹不存在则创建, 若文件夹已存在且不为空，则终止程序且给出提示
        # 检查文件夹是否存在
        if rank == 0:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"文件夹 {save_folder} 已成功创建。")
            else:
                # 如果文件夹存在，检查是否为空
                if os.listdir(save_folder):  # os.listdir() 返回文件夹中的文件列表
                    sys.stderr.write(f"Error: 数据文件夹 {save_folder} 已存在且不为空，所有进程将被终止。\n")
                    comm.Abort(1)  # 终止所有进程
                else:
                    print(f"文件夹 {save_folder} 已存在但为空，可以继续使用. current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            time.sleep(3)
        # 确保所有进程都同步等 rank 0 完成检查
        comm.Barrier()
        # 在初始化时确定需要保存的随机粒子索引，后续使用相同的粒子
        self.selected_indices = None
        print(f"Rank {rank} 完成文件夹{save_folder}检查. current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    def initialize_particles_to_dump(self, bunch_obj):
        """
        确定需要保存的粒子序号，只在初始化时调用一次。
        :param bunch_obj: FFAG_Bunch 对象，表示当前的粒子束。
        """
        total_particles = bunch_obj.LocalBunch.shape[0]

        # 如果粒子总数少于需要保存的粒子数，则选取所有粒子
        if total_particles <= self.num_particles_to_dump:
            self.selected_indices = np.arange(total_particles)
        else:
            # 随机选取指定数量的粒子索引
            self.selected_indices = np.random.choice(total_particles, self.num_particles_to_dump, replace=False)

    def check_and_dump(self, step_i, time_i, bunch_obj):
        """
        检查当前步数是否满足 Dump 条件，如果满足则进行 Dump 操作。
        :param step_i: int, 当前步数。
        :param bunch_obj: FFAG_Bunch 对象，表示当前的粒子束。
        """
        if step_i % self.step_interval == 0:
            self.dump_particles(bunch_obj, step_i, time_i)

    def dump_particles(self, bunch_obj, step_i, time_i):
        """
        执行粒子的 Dump 操作，保存已选定的 n 个粒子的信息。
        :param bunch_obj: FFAG_Bunch 对象。
        :param time_i: int, 当前时间或步数，用于标记当前的粒子状态。
        """
        # 确保已初始化粒子索引
        if self.selected_indices is None:
            self.initialize_particles_to_dump(bunch_obj)

        # 根据选定的索引保存对应粒子的信息
        particles_to_dump = bunch_obj.LocalBunch[self.selected_indices, :]

        # 获取当前 MPI 线程号
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # 遍历要保存的粒子，并将其信息追加到对应文件中
        for i, particle in enumerate(particles_to_dump):
            # 获取粒子的 Global_ID，作为文件名的一部分
            global_id = int(particle[bunch_obj.BunchAttribute.Attribute['Global_ID']])

            file_name = os.path.join(self.save_folder, f"particle_{global_id}_rank_{rank}.npz")

            # 将粒子在当前步的状态追加写入文件
            append_to_npz(file_name, np.concatenate((particle, [time_i])))


class StepDumpBunch:
    def __init__(self, step_interval, save_folder="dump_data"):
        """
         StepDumpBunch类，用于在 RK 积分过程中每隔一定的 step 进行一次束团 Dump操作。
        :param step_interval: int, 每隔多少步进行一次 Dump。
        :param save_folder: str, 保存文件的文件夹路径，不包含 rank，rank 号会自动添加为子文件夹。
        """
        self.step_interval = step_interval
        self.save_folder = save_folder

        # 获取 MPI 线程号
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # 创建保存路径，如果文件夹不存在则创建, 若文件夹已存在且不为空，则终止程序且给出提示
        if rank == 0:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"文件夹 {save_folder} 已成功创建。")
            else:
                if os.listdir(save_folder):  # 检查文件夹是否为空
                    sys.stderr.write(f"Error: 数据文件夹 {save_folder} 已存在且不为空，所有进程将被终止。\n")
                    comm.Abort(1)  # 终止所有进程
                else:
                    print(
                        f"文件夹 {save_folder} 已存在但为空，可以继续使用。current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            time.sleep(0.5)

        # 确保所有进程都同步等 rank 0 完成检查
        comm.Barrier()
        print(f"Rank {rank} 完成文件夹检查, current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    def check_and_dump(self, step_i, time_i, bunch_obj):
        """
        检查当前步数是否满足 Dump 条件，如果满足则进行 Dump 操作。
        :param step_i: int, 当前步数。
        :param bunch_obj: FFAG_Bunch 对象，表示当前的粒子束。
        """
        if step_i % self.step_interval == 0:
            self.dump_bunch(bunch_obj, step_i, time_i)

    def dump_bunch(self, bunch_obj, step_i, time_i):
        """
        执行粒子的 Dump 操作，保存当前的粒子束信息。
        :param bunch_obj: FFAG_Bunch 对象。
        :param step_i: int, 当前步数，用于标记当前的粒子状态。
        """
        # 获取当前 MPI 线程号
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # 获取粒子数据
        LocalBunch = bunch_obj.LocalBunch

        # 筛选存活的粒子
        survive_flag_idx = bunch_obj.BunchAttribute.Attribute['Survive']
        survived_particles = LocalBunch[LocalBunch[:, survive_flag_idx] == 1]

        # 如果没有存活的粒子，则跳过
        if len(survived_particles) == 0:
            return

        # 创建文件名
        filename = f"step_{step_i}_rank_{rank}.npz"
        filepath = os.path.join(self.save_folder, filename)

        # 保存存活粒子数据到文件，使用 npz 格式
        particles_with_time = np.hstack((survived_particles, np.full((survived_particles.shape[0], 1), time_i)))
        append_to_npz(filepath, particles_with_time)


class PositionDump:
    def __init__(self, target_azimuth_angle_deg, save_folder="dump_data", dump_turns=None):
        """
        PositionDump类，用于粒子穿越给定方位角时进行 Dump 操作。
        :param target_azimuth_angle: float, 目标方位角 (rad)。
        :param dump_turns: list or np.ndarray, 指定需要进行 Dump 操作的圈数。
        """
        self.target_azimuth_angle = np.deg2rad(target_azimuth_angle_deg) % (2 * np.pi)  # 将方位角归一化到 [0, 2π) 范围内
        self.dump_turns = dump_turns  # 用户可以提供任意圈数的列表
        self.folder_path = save_folder

        # 获取 MPI 线程号
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        # 创建保存路径，如果文件夹不存在则创建, 若文件夹已存在且不为空，则终止程序且给出提示
        if rank == 0:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"文件夹 {save_folder} 已成功创建。")
            else:
                if os.listdir(save_folder):  # 检查文件夹是否为空
                    sys.stderr.write(f"Error: 数据文件夹 {save_folder} 已存在且不为空，所有进程将被终止。\n")
                    comm.Abort(1)  # 终止所有进程
                else:
                    print(
                        f"文件夹 {save_folder} 已存在但为空，可以继续使用。current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            time.sleep(0.5)

        # 确保所有进程都同步等 rank 0 完成检查
        comm.Barrier()
        print(f"Rank {rank} 完成文件夹检查, current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        # # 检查并创建文件夹
        # if not os.path.exists(self.folder_path):
        #     os.makedirs(self.folder_path)
        #     print(f"文件夹 {self.folder_path} 已成功创建。")
        # else:
        #     # 如果文件夹已存在，检查是否为空
        #     if os.listdir(self.folder_path):  # os.listdir() 返回文件夹中的文件列表
        #         print(f"警告: 文件夹 {self.folder_path} 已存在且不为空！")
        #         response = input("是否清空文件夹内容？(y/n): ").strip().lower()
        #         if response == 'y':
        #             # 清空文件夹内容
        #             for filename in os.listdir(self.folder_path):
        #                 file_path = os.path.join(self.folder_path, filename)
        #                 try:
        #                     if os.path.isfile(file_path):
        #                         os.remove(file_path)
        #                     elif os.path.isdir(file_path):
        #                         os.rmdir(file_path)  # 只适用于空文件夹
        #                 except Exception as e:
        #                     print(f"无法删除文件 {file_path}: {e}")
        #             print(f"文件夹 {self.folder_path} 已被清空。")
        #         else:
        #             print(f"保留文件夹 {self.folder_path} 的内容。")
        #     else:
        #         print(f"文件夹 {self.folder_path} 已存在但为空，可以继续使用。")
        #
        # print(f"PositionDump 初始化完成，文件夹路径: {self.folder_path}")

    def check_azimuth_crossing(self, bunch_obj):
        """
        检查粒子是否穿越了指定的目标方位角，并且检查是否满足圈数筛选条件
        每个线程独立运行，独立检查每个线程的local bunch
        返回一个1维ndarray，满足条件的粒子返回圈数，不满足条件的粒子返回-1
        使用ndarray向量化运算，避免使用for循环
        """
        pre_fi = bunch_obj.Pre_steps[:, 3]  # Pre-step的方位角
        post_fi = bunch_obj.Post_steps[:, 3]  # Post-step的方位角
        survive_flag = bunch_obj.Post_steps[:, 4]  # Post-step的存活标志

        # 获取粒子的方位角
        pre_fi_valid = pre_fi % (2 * np.pi)  # 将方位角归一化到 [0, 2π) 范围内
        post_fi_valid = post_fi % (2 * np.pi)

        # 获取有效粒子的圈数
        post_turn_valid = post_fi // (2 * np.pi)

        # 初始化结果数组，默认所有粒子不满足条件，即返回值为 -1
        result_turns = np.full(len(pre_fi), -1)

        # pre_fi_valid < target_fi, post_fi_valid >= target_fi 表示粒子从 Pre_steps 到 Post_steps 之间跨越了目标方位角
        # survive_flag == 1 表示粒子存活
        # np.isin(post_turn_valid, self.dump_turns) 表示粒子的圈数在指定的 dump_turns 中
        crossing_mask_valid = (
                (pre_fi_valid < self.target_azimuth_angle) &
                (post_fi_valid >= self.target_azimuth_angle) &
                (survive_flag == 1)
        )

        # 将穿越目标方位角的粒子圈数更新到结果数组中
        result_turns[crossing_mask_valid] = post_turn_valid[crossing_mask_valid]

        return result_turns  # 返回一个包含圈数的列表，未满足条件的粒子返回 -1

    def dump_crossing_particles_to_files(self, bunch_obj, result_turns, time_i):
        """
        将穿越目标方位角的粒子数据保存为 .npz 文件。每个方位角、圈数和 MPI 线程号对应一个文件。
        :param bunch_obj: FFAG_Bunch 对象，包含粒子信息
        :param result_turns: ndarray，表示粒子的穿越给定方位角信息和圈数信息
        """
        # 获取当前 MPI 线程号
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # 获取粒子信息
        LocalBunch = bunch_obj.LocalBunch

        # 筛选出所有穿越目标方位角的粒子（result_turns != -1）
        valid_particles_mask = result_turns != -1
        valid_particles_data = LocalBunch[valid_particles_mask, :]  # 符合条件的粒子数据
        valid_turns = result_turns[valid_particles_mask]  # 符合条件的圈数信息

        # 如果没有符合条件的粒子，则跳过 dump
        if len(valid_turns) == 0:
            return

        # 遍历所有符合条件的粒子，按圈数分别进行 dump
        for turn in np.unique(valid_turns):
            # 筛选出当前圈数的粒子
            turn_mask = valid_turns == turn
            turn_particles_data = valid_particles_data[turn_mask, :]

            # 创建文件名
            filename = f"crossing_angle_{np.rad2deg(self.target_azimuth_angle):.1f}_turn_{int(turn)}_rank_{rank}.npz"
            filepath = os.path.join(self.folder_path, filename)

            # 将粒子数据保存为 .npz 格式
            particles_with_time = np.hstack((turn_particles_data, np.full((turn_particles_data.shape[0], 1), time_i)))
            append_to_npz(filepath, particles_with_time)


    def check_and_dump(self, step_i, time_i, bunch_obj):
        """
        检查粒子是否穿越指定的目标方位角，如果满足条件则保存相关数据。

        :param step_i: int, 当前模拟步数。
        :param bunch_obj: FFAG_Bunch 对象，包含粒子数据。
        """
        # 检查粒子是否穿越目标方位角，并返回圈数信息
        result_turns = self.check_azimuth_crossing(bunch_obj)

        # 如果有粒子满足穿越条件，则进行数据保存
        if np.any(result_turns != -1):  # 如果有满足条件的粒子
            self.dump_crossing_particles_to_files(bunch_obj, result_turns, time_i)


class Dumps:
    def __init__(self):
        """
        Dumps 类，用于管理多个独立的 Dump 实例。
        """
        self.dumps = []  # 存储所有 dump 实例

    def add_dump(self, dump_obj):
        """
        添加一个 dump 实例。
        :param dump_obj: 一个 dump 实例，例如 StepDump, StepDumpBunch 或 PositionDump。
        """
        self.dumps.append(dump_obj)

    def check_and_dump(self, step_i, time_i, bunch_obj):
        """
        遍历所有 dump 实例，分别调用其 check_and_dump 方法进行数据存储。
        :param step_i: int, 当前步数。
        :param time_i: float, 当前时间。
        :param bunch_obj: FFAG_Bunch 对象，表示当前的粒子束。
        """
        for dump in self.dumps:
            dump.check_and_dump(step_i, time_i, bunch_obj)

    def list_dumps(self):
        """
        列出所有已添加的 dump 实例及其类型。
        """
        for i, dump in enumerate(self.dumps):
            print(f"Dump {i + 1}: {dump.__class__.__name__}")


def merge_files_in_folder(folder_path):
    """
    合并文件夹中的子文件，将文件名中除了 rank 不同但其他部分相同的文件合并为一个文件。
    合并后的文件保存在输入文件夹下的子文件夹 'merged_files' 中。
    """
    file_dict = {}
    merged_folder = os.path.join(folder_path, "merged_files")  # 合并后的文件存放子文件夹

    # 创建存放合并文件的子文件夹
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.npz'):
            # 移除文件名中的 rank 信息，作为 key 存入字典
            file_key = '_'.join(filename.split('_')[:-1])  # 移除 rank 部分
            if file_key not in file_dict:
                file_dict[file_key] = []
            file_dict[file_key].append(os.path.join(folder_path, filename))

    # 对于字典中的每组文件，进行合并
    for file_key, file_list in file_dict.items():
        merged_data = []
        output_file = f"{file_key}_merged.csv"  # 生成合并后的文件名
        output_filepath = os.path.join(merged_folder, output_file)

        # 遍历每个文件并加载数据
        for file in file_list:
            try:
                # 读取 npz 文件中键为 'particles' 的数据
                with np.load(file) as npz_file:
                    data = npz_file['particles']
                    merged_data.append(data)
            except Exception as e:
                print(f"Error loading file {file}: {e}")
                continue

        bunch_attribute = FFAG_ManageBunchAttribute()
        # 按 FFAG_ManageBunchAttribute 中的定义顺序和格式生成保存格式
        attribute_names = list(bunch_attribute.Attribute.keys())  # 属性名列表（按顺序）
        column_formats = [bunch_attribute.AttributeFormat[attr] for attr in attribute_names] + ['%.8e']  # 属性对应的格式列表

        # 将所有数据合并并保存为一个文件
        if merged_data:  # 确保有数据可以合并
            merged_data = np.vstack(merged_data)

            # 获取 Global_ID 列的索引
            global_id_index = list(bunch_attribute.Attribute.keys()).index('Global_ID')

            # 检查 Global_ID 列是否有多个不同的值
            if len(np.unique(merged_data[:, global_id_index])) > 1:
                # 按照 Global_ID 列排序（从小到大）
                sorted_indices = np.argsort(merged_data[:, global_id_index])  # 获取排序后的索引
                merged_data_sorted = merged_data[sorted_indices]  # 按照索引对数据进行排序
                print(f"Sorted data based on Global_ID.")
            else:
                merged_data_sorted = merged_data  # 如果 Global_ID 都相同，不进行排序
                print(f"Global_ID values are the same, skipping sorting.")

            # 使用生成的格式保存文件
            np.savetxt(output_filepath, merged_data_sorted, delimiter=',', fmt=tuple(column_formats))

            print(f"Merged {len(file_list)} files into {output_filepath}")
        else:
            print(f"No valid data found for {file_key}, skipping merge.")


def append_to_npz(filepath, new_data):
    """
    追加新数据到 .npz 文件。如果文件已存在，加载原有数据并合并后保存；否则直接保存新数据。

    :param filepath: str, .npz 文件路径。
    :param new_data: ndarray, 要追加的数据。
    """
    if os.path.exists(filepath):
        # 加载已有数据
        existing_data = np.load(filepath)
        existing_particles = existing_data['particles']

        # 合并数据
        combined_data = np.vstack((existing_particles, new_data))
    else:
        # 文件不存在时，直接保存新数据
        combined_data = new_data

    # 重新保存合并后的数据
    np.savez_compressed(filepath, particles=combined_data)
    # print(f"Data saved to {filepath}. Total particles: {combined_data.shape[0]}")

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Merge files in a folder.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing files to merge.")
    args = parser.parse_args()

    # 调用合并函数
    merge_files_in_folder(args.folder_path)

# # 测试程序
# if __name__ == "__main__":
#     # 初始化 MPI
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#
#     # 定义一个模拟的粒子分布
#     np.random.seed(int(MPI.Wtime() * 1000) + rank)  # 使用当前时间和rank作为随机种子
#     ParticleNum = 1000
#     ParticlesDistribution = np.column_stack((
#         np.random.random(ParticleNum) * 10,  # r
#         np.random.random(ParticleNum),  # vr
#         np.random.random(ParticleNum) * 10,  # z
#         np.random.random(ParticleNum),  # vz
#         np.random.random(ParticleNum) * 2 * np.pi,  # fi (方位角)
#         np.random.random(ParticleNum) * 100,  # Ek
#         np.random.random(ParticleNum) * 5,  # inj_t
#         np.ones(ParticleNum),  # survive flag
#         np.random.random(ParticleNum) * 2 * np.pi,  # RF_phase
#         np.zeros(ParticleNum),  # Esc_r (initially zero)
#         np.zeros(ParticleNum),  # Esc_z (initially zero)
#         np.zeros(ParticleNum),  # Esc_fi (initially zero)
#         np.zeros(ParticleNum),  # Bunch_ID
#         np.zeros(ParticleNum),  # Local_ID (will be set by the class)
#         np.zeros(ParticleNum)  # Global_ID (will be set by the class)
#     ))
#
#     # 创建一个 FFAG_Bunch 实例
#     bunch = FFAG_Bunch(ParticlesDistribution)
#
#     # 定义 StepDump 实例
#     step_dump = StepDump(step_interval=100, num_particles_to_dump=10, save_folder=f"./step_dump")
#
#     # 模拟粒子轨迹积分的过程，执行多个 step 更新粒子位置
#     num_steps = 2000  # 模拟 2000 步
#     for step in range(num_steps):
#         # 更新 Pre-steps (积分之前)
#         bunch.UpdatePreSteps()
#
#         # 模拟粒子轨迹积分 (简单模拟粒子方位角变化)
#         bunch.LocalBunch[:, 4] += 0.01  # 每步让粒子的方位角增加
#
#         # 更新 Post-steps (积分之后)
#         bunch.UpdatePostSteps()
#
#         # 每隔一定步数进行粒子数据的 Dump
#         step_dump.check_and_dump(step, bunch)
#
#     # print(f"Simulation completed on rank {rank}")
