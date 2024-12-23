from mpi4py import MPI
import numpy as np
from FFAG_track import FFAG_MPI
import copy
from FFAG_SC import find_two_points_non_uniform_vect_njit, DistributeChargeNjit
from FFAG_ParasAndConversion import FFAG_GlobalParameters


class FFAG_ManageBunchAttribute:
    def __init__(self):
        # 初始化属性字典
        self.Attribute = dict()
        self.Attribute['r'] = int(0)
        self.Attribute['vr'] = int(1)
        self.Attribute['z'] = int(2)
        self.Attribute['vz'] = int(3)
        self.Attribute['fi'] = int(4)
        self.Attribute['Ek'] = int(5)
        self.Attribute['inj_t'] = int(6)
        self.Attribute['Survive'] = int(7)
        self.Attribute['RF_phase'] = int(8)
        self.Attribute['Esc_r'] = int(9)
        self.Attribute['Esc_z'] = int(10)
        self.Attribute['Esc_fi'] = int(11)
        self.Attribute['Bunch_ID'] = int(12)
        self.Attribute['Local_ID'] = int(13)
        self.Attribute['Global_ID'] = int(14)

        self.AttributeFormat = dict()  # 新增的属性格式字典
        # 定义属性的保存格式
        self.AttributeFormat['r'] = '%.8e'
        self.AttributeFormat['vr'] = '%.8e'
        self.AttributeFormat['z'] = '%.8e'
        self.AttributeFormat['vz'] = '%.8e'
        self.AttributeFormat['fi'] = '%.8e'
        self.AttributeFormat['Ek'] = '%.8e'
        self.AttributeFormat['inj_t'] = '%.6e'
        self.AttributeFormat['Survive'] = '%d'
        self.AttributeFormat['RF_phase'] = '%.6e'
        self.AttributeFormat['Esc_r'] = '%.6e'
        self.AttributeFormat['Esc_z'] = '%.6e'
        self.AttributeFormat['Esc_fi'] = '%.6e'
        self.AttributeFormat['Bunch_ID'] = '%d'
        self.AttributeFormat['Local_ID'] = '%d'
        self.AttributeFormat['Global_ID'] = '%d'

    def get_num_attributes(self):
        """
        获取当前属性的个数，即Bunch矩阵的列数。
        """
        return len(self.Attribute)

    def get_attribute_names(self):
        """
        获取所有属性名称的列表。
        """
        return list(self.Attribute.keys())


class FFAG_Bunch:

    def __init__(self, ParticlesDistribution, marcosize=1):
        # marcosize is the scaling factor between
        # the simulated macro particles and the actual particle count.
        self.StaticBunchGlobal = copy.deepcopy(ParticlesDistribution)

        self.BunchAttribute = FFAG_ManageBunchAttribute()

        # Set the Global ID
        self.TotalParticleNum = np.size(self.StaticBunchGlobal, 0)
        self.marcosize = marcosize
        self.StaticBunchGlobal[:, self.BunchAttribute.Attribute['Global_ID']] = np.arange(self.TotalParticleNum)

        # StaticBunch represents the static initial distribution \
        # that will remain unchanged
        # GlobalBunch and LocalBunch are dynamic and will change in each step
        # (r, vr), (z, vz), (fi, dfidt), (t_inj, survive_flag),
        # (rf_phase, Esc_r, Esc_z, Esc_fi)
        # (Bunch_ID, Local_ID, Global_ID)

        # divide particles for threads
        PIDGlobal = self.StaticBunchGlobal[:, self.BunchAttribute.Attribute['Global_ID']]
        PIDLocal = FFAG_MPI().DivideVariables(PIDGlobal)
        ParticlesDistribution_local = self.StaticBunchGlobal[PIDLocal, :]

        # generate local Static ini bunch
        self.StaticBunchLocal = copy.deepcopy(ParticlesDistribution_local)
        self.TotalParticleNum_Local = np.size(self.StaticBunchLocal, 0)
        # generate the local id
        self.StaticBunchLocal[:, self.BunchAttribute.Attribute['Local_ID']] = np.arange(self.TotalParticleNum_Local)

        self.GlobalBunch = copy.deepcopy(self.StaticBunchGlobal)
        self.LocalBunch = copy.deepcopy(self.StaticBunchLocal)

        BunchMatrixColNum = self.BunchAttribute.get_num_attributes()  # Bunch矩阵的列数 Bunch的属性数
        self.LocalLostBunch = np.zeros((0, BunchMatrixColNum))  # 初始化束损Bunch矩阵
        self.GlobalLostBunch = np.zeros((0, BunchMatrixColNum))  # 初始化束损Bunch矩阵

        # Add Pre_step and Post_step for tracking (x, y, z, fi)
        self.Pre_steps = np.zeros((self.TotalParticleNum_Local, 6))  # Pre-step positions (x, y, z, fi, survive_flag, GlobalID)
        self.Post_steps = np.zeros((self.TotalParticleNum_Local, 6))  # Post-step positions (x, y, z, fi, survive_flag, GlobalID)

        self.point_x = None
        self.point_y = None
        self.point_z = None

        self.xmin_Local = None
        self.xmax_Local = None
        self.ymin_Local = None
        self.ymax_Local = None
        self.zmin_Local = None
        self.zmax_Local = None

        self.xmin_Global = None
        self.xmax_Global = None
        self.ymin_Global = None
        self.ymax_Global = None
        self.zmin_Global = None
        self.zmax_Global = None

        self.Xgrid = None
        self.Ygrid = None
        self.Zgrid = None
        self.Xmatrix = None
        self.Ymatrix = None
        self.Zmatrix = None

        self.charge_distribution_local = None
        self.charge_distribution_global = None

        # statistics of ParticleNum
        # TotalParticleNum
        # --1. InjectedNum
        # ----1.1 SurvivedNum
        # ----1.2 LostNum
        # --2. UnInjectNum
        # RemainedNum = UnInjectNum + SurvivedNum = TotalNum - LostNum
        self.InjectedNum = 0
        self.UnInjectNum = self.TotalParticleNum  # 未注入的粒子初始为总粒子数
        self.SurvivedNum = 0
        self.LostNum = 0
        self.RemainedNum = self.TotalParticleNum  # 剩余粒子初始为总粒子数

        self.InjectedNum_Local = 0
        self.UnInjectNum_Local = self.TotalParticleNum_Local
        self.SurvivedNum_Local = 0
        self.LostNum_Local = 0
        self.RemainedNum_Local = self.TotalParticleNum_Local

        self.CurrentTurn = 0

    def UpdateAndInjectParticles(self, t_threshold):
        """
        将符合注入条件的粒子注入系统，并更新相关粒子数统计，每个step调用一次。
        :param t_threshold: 当前时间阈值
        """
        inj_time_idx = self.BunchAttribute.Attribute['inj_t']
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        # 跟踪过程中，只计算survive_flag == 1 的粒子
        # 未注入的粒子, 其survive_flag = 0, 达到注入时间时, survive_flag变为1,
        # 损失时survive_flag变为-1
        # 当前存活粒子数 ： 所有survive_flag=1的粒子
        # 当前束损粒子数 ： 所有survive_flag=-1的粒子
        # 已注入粒子数 ： current time > inj time
        # 未注入粒子数 ： current time < inj time
        # 找到符合条件的粒子 (注入时间 <= 当前时间阈值，且survive_flag == 0)
        # 对本地线程的Local Bunch进行操作
        waiting_to_inject_local_mask = (self.LocalBunch[:, inj_time_idx] <= t_threshold) & \
                                       (self.LocalBunch[:, survive_flag_idx] == 0)
        num_to_inject_local = np.sum(waiting_to_inject_local_mask)

        if num_to_inject_local > 0:
            self.LocalBunch[waiting_to_inject_local_mask, survive_flag_idx] = 1
            self.InjectedNum_Local += num_to_inject_local
            self.UnInjectNum_Local -= num_to_inject_local
            self.SurvivedNum_Local += num_to_inject_local

        self.UpdateParticleNumGlobal()

        return 0

    # @profile
    def UpdateParticleNumGlobal(self):
        """
        使用MPI收集所有进程的本地粒子数，并更新全局粒子数统计。
        """
        # 初始化MPI通信
        comm = MPI.COMM_WORLD

        # 汇总本地粒子数信息
        global_injected_num = comm.reduce(self.InjectedNum_Local, op=MPI.SUM, root=0)
        global_uninject_num = comm.reduce(self.UnInjectNum_Local, op=MPI.SUM, root=0)
        global_survived_num = comm.reduce(self.SurvivedNum_Local, op=MPI.SUM, root=0)
        global_lost_num = comm.reduce(self.LostNum_Local, op=MPI.SUM, root=0)

        # 仅在主进程（rank == 0）更新全局统计量
        if comm.rank == 0:
            self.InjectedNum = global_injected_num
            self.UnInjectNum = global_uninject_num
            self.SurvivedNum = global_survived_num
            self.LostNum = global_lost_num
            self.RemainedNum = self.TotalParticleNum - self.LostNum

        # 同步全局统计信息到所有进程
        self.InjectedNum = comm.bcast(self.InjectedNum, root=0)
        self.UnInjectNum = comm.bcast(self.UnInjectNum, root=0)
        self.SurvivedNum = comm.bcast(self.SurvivedNum, root=0)
        self.LostNum = comm.bcast(self.LostNum, root=0)
        self.RemainedNum = comm.bcast(self.RemainedNum, root=0)

    def UpdatePreSteps(self):
        """
        每个step进行积分前调用，读取LocalBunch矩阵中的坐标，更新PreSteps。
        """
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        # 只更新存活的粒子
        valid_particles_mask = self.LocalBunch[:, survive_flag_idx] == 1

        # 更新 Pre_steps 为当前的粒子位置
        self.Pre_steps[valid_particles_mask, 0] = self.LocalBunch[
                                                      valid_particles_mask, self.BunchAttribute.Attribute['r']] * \
                                                  np.cos(self.LocalBunch[
                                                             valid_particles_mask, self.BunchAttribute.Attribute['fi']])
        self.Pre_steps[valid_particles_mask, 1] = self.LocalBunch[
                                                      valid_particles_mask, self.BunchAttribute.Attribute['r']] * \
                                                  np.sin(self.LocalBunch[
                                                             valid_particles_mask, self.BunchAttribute.Attribute['fi']])
        self.Pre_steps[valid_particles_mask, 2] = self.LocalBunch[
            valid_particles_mask, self.BunchAttribute.Attribute['z']]
        self.Pre_steps[valid_particles_mask, 3] = self.LocalBunch[
            valid_particles_mask, self.BunchAttribute.Attribute['fi']]
        self.Pre_steps[valid_particles_mask, 4] = self.LocalBunch[
            valid_particles_mask, survive_flag_idx]  # 更新 survive_flag
        self.Pre_steps[valid_particles_mask, 5] = self.LocalBunch[
            valid_particles_mask, self.BunchAttribute.Attribute['Global_ID']]  # 更新 survive_flag

    def UpdatePostSteps(self):
        """
        每个step进行积分后调用(在DeleteParticles后)，读取LocalBunch矩阵中的坐标，更新PostSteps。
        """
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        # 只更新存活的粒子
        valid_particles_mask = self.LocalBunch[:, survive_flag_idx] == 1

        # 更新 Post_steps 为当前的粒子位置
        self.Post_steps[valid_particles_mask, 0] = self.LocalBunch[
                                                       valid_particles_mask, self.BunchAttribute.Attribute['r']] * \
                                                   np.cos(self.LocalBunch[
                                                              valid_particles_mask, self.BunchAttribute.Attribute[
                                                                  'fi']])
        self.Post_steps[valid_particles_mask, 1] = self.LocalBunch[
                                                       valid_particles_mask, self.BunchAttribute.Attribute['r']] * \
                                                   np.sin(self.LocalBunch[
                                                              valid_particles_mask, self.BunchAttribute.Attribute[
                                                                  'fi']])
        self.Post_steps[valid_particles_mask, 2] = self.LocalBunch[
            valid_particles_mask, self.BunchAttribute.Attribute['z']]
        self.Post_steps[valid_particles_mask, 3] = self.LocalBunch[
            valid_particles_mask, self.BunchAttribute.Attribute['fi']]
        self.Post_steps[valid_particles_mask, 4] = self.LocalBunch[
            valid_particles_mask, survive_flag_idx]  # 更新 survive_flag
        self.Post_steps[valid_particles_mask, 5] = self.LocalBunch[
            valid_particles_mask, self.BunchAttribute.Attribute['Global_ID']]

    def UpdateGlobalBunch(self):
        """
        使用MPI收集每个进程的本地粒子束并更新全局粒子束，并根据GlobalBunch更新全局粒子数统计。
        """
        # Initialize MPI
        comm = MPI.COMM_WORLD
        # 使用allgather收集所有进程的LocalBunch
        LocalBunchGather = comm.allgather(self.LocalBunch)
        # 将所有本地粒子束拼接成全局粒子束
        self.GlobalBunch = np.concatenate(LocalBunchGather, axis=0)

        ##  更新全局粒子数
        # 根据GlobalBunch统计全局粒子数
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']
        # 统计已注入的粒子数（survive_flag == 1 或 -1 表示已注入的粒子）
        self.InjectedNum = np.sum(self.GlobalBunch[:, survive_flag_idx] != 0)
        # 统计未注入的粒子数（survive_flag == 0）
        self.UnInjectNum = np.sum(self.GlobalBunch[:, survive_flag_idx] == 0)
        # 统计存活的粒子数（survive_flag == 1）
        self.SurvivedNum = np.sum(self.GlobalBunch[:, survive_flag_idx] == 1)
        # 统计丢失的粒子数（survive_flag == -1）
        self.LostNum = np.sum(self.GlobalBunch[:, survive_flag_idx] == -1)
        # 剩余的粒子数
        self.RemainedNum = self.TotalParticleNum - self.LostNum

    def DeleteParticles(self, DeleteIndexLocal):
        """
        删除本地粒子束中的指定粒子，并将其移动到LocalLostBunch中。损失的粒子被标记并更新。
        :param DeleteIndexLocal: 一个索引数组，指示哪些粒子需要被删除。
        """
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        # 1. 将要删除的粒子添加到 LocalLostBunch中
        self.LocalLostBunch = np.row_stack((self.LocalLostBunch, self.LocalBunch[DeleteIndexLocal, :]))

        # 2. 更新这些粒子的状态，将 survive_flag 设为 -1 表示损失
        self.LocalBunch[DeleteIndexLocal, survive_flag_idx] = -1

        # 3. 将这些粒子的关键属性（如位置、速度等）标记为 NaN，表示它们已被移除
        self.LocalBunch[DeleteIndexLocal, 0:6] = np.nan  # 假设 0:6 是粒子的状态变量（位置、速度等）

        # 4. 可以选择在此处更新损失的粒子数
        num_lost = len(DeleteIndexLocal)
        self.LostNum_Local += num_lost
        self.SurvivedNum_Local -= num_lost

        self.UpdateParticleNumGlobal()

    def GetMeanEk(self):
        """
        获取GlobalBunch中，存活粒子的平均动能Ek。
        如果没有存活粒子，则返回静态注入粒子束中的平均Ek。
        """
        ek_idx = self.BunchAttribute.Attribute['Ek']
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        if self.InjectedNum == 0 or self.SurvivedNum == 0:
            # 如果没有已注入或存活的粒子，返回静态粒子束中的平均Ek
            MeanEk = np.mean(self.StaticBunchGlobal[:, ek_idx])
        else:
            # 筛选出survive_flag == 1的存活粒子
            survived_particles_mask = self.GlobalBunch[:, survive_flag_idx] == 1

            # 获取存活粒子的动能
            survived_ek = self.GlobalBunch[survived_particles_mask, ek_idx]

            # 计算存活粒子的平均动能
            if len(survived_ek) > 0:
                MeanEk = np.mean(survived_ek)
            else:
                # 防止没有存活粒子时的异常情况
                MeanEk = 0

        return MeanEk

    # def GetAllCoordinates(self):
    #     """
    #     返回Bunch中所有存活粒子的x, y, z笛卡尔坐标，以n*3的ndarray形式返回。
    #     """
    #     # 获取属性的索引
    #     r_idx = self.BunchAttribute.Attribute['r']
    #     fi_idx = self.BunchAttribute.Attribute['fi']
    #     z_idx = self.BunchAttribute.Attribute['z']
    #     survive_flag_idx = self.BunchAttribute.Attribute['Survive']
    #
    #     # 筛选出所有存活的粒子（survive_flag == 1），排除已损失或未注入的粒子
    #     valid_particles_mask = self.GlobalBunch[:, survive_flag_idx] == 1
    #
    #     # 提取有效粒子的 r, fi, z 坐标
    #     r_polar = self.GlobalBunch[valid_particles_mask, r_idx]
    #     fi_polar = self.GlobalBunch[valid_particles_mask, fi_idx]
    #     z_cart = self.GlobalBunch[valid_particles_mask, z_idx]
    #
    #     # 将极坐标 (r, fi) 转换为笛卡尔坐标 (x, y)
    #     x_cart = r_polar * np.cos(fi_polar)
    #     y_cart = r_polar * np.sin(fi_polar)
    #
    #     # 将 x, y, z 坐标组合为 n*3 的 ndarray
    #     coordinates = np.column_stack((x_cart, y_cart, z_cart))
    #
    #     return coordinates

    def GetLocalCoordinates(self):
        """
        返回本地 Bunch 中所有存活粒子的 x, y, z 笛卡尔坐标（n, 3 的 ndarray），
        以及 LocalID 和 GlobalID（分别为 n, 的整数类型的 ndarray）。
        """
        # 获取属性的索引
        r_idx = self.BunchAttribute.Attribute['r']
        fi_idx = self.BunchAttribute.Attribute['fi']
        z_idx = self.BunchAttribute.Attribute['z']
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']
        local_id_idx = self.BunchAttribute.Attribute['Local_ID']
        global_id_idx = self.BunchAttribute.Attribute['Global_ID']

        # 筛选出本地存活的粒子（survive_flag == 1），排除已损失或未注入的粒子
        valid_particles_mask = self.LocalBunch[:, survive_flag_idx] == 1

        # 提取有效粒子的 r, fi, z 坐标
        r_polar = self.LocalBunch[valid_particles_mask, r_idx]
        fi_polar = self.LocalBunch[valid_particles_mask, fi_idx]
        z_cart = self.LocalBunch[valid_particles_mask, z_idx]

        # 将极坐标 (r, fi) 转换为笛卡尔坐标 (x, y)
        x_cart = r_polar * np.cos(fi_polar)
        y_cart = r_polar * np.sin(fi_polar)

        # 提取有效粒子的 LocalID 和 GlobalID，并将它们转换为整数
        local_id = self.LocalBunch[valid_particles_mask, local_id_idx].astype(int)
        global_id = self.LocalBunch[valid_particles_mask, global_id_idx].astype(int)

        # 将 x, y, z 组合为 (n, 3) 的 ndarray
        coordinates = np.column_stack((x_cart, y_cart, z_cart))

        # 返回笛卡尔坐标和整数类型的 LocalID、GlobalID
        return coordinates, local_id, global_id

    def Get_SC_Grid_para_Local(self):
        """
        获取当前存活粒子的边界参数，并将极坐标转换为笛卡尔坐标。
        更新 self.xmin_Local, self.xmax_Local, self.ymin_Local,
        self.ymax_Local, self.zmin_Local, self.zmax_Local
        """
        survive_flag_idx = self.BunchAttribute.Attribute['Survive']

        # 筛选出存活粒子（survive_flag == 1）
        survived_mask = self.LocalBunch[:, survive_flag_idx] == 1
        if not np.any(survived_mask):
            # 如果没有存活粒子，将边界参数设为 None
            self.xmin_Local = None
            self.xmax_Local = None
            self.ymin_Local = None
            self.ymax_Local = None
            self.zmin_Local = None
            self.zmax_Local = None
            return

        # 提取局部存活粒子的 r, fi 和 z 坐标
        r_polar = self.LocalBunch[survived_mask, self.BunchAttribute.Attribute['r']]  # 粒子的径向坐标
        fi_polar = self.LocalBunch[survived_mask, self.BunchAttribute.Attribute['fi']]  # 粒子的方位角坐标
        z_cart = self.LocalBunch[survived_mask, self.BunchAttribute.Attribute['z']]  # 粒子的纵向坐标（z 方向）

        # 将极坐标转换为笛卡尔坐标
        x_cart = r_polar * np.cos(fi_polar)
        y_cart = r_polar * np.sin(fi_polar)

        # 计算 x, y 和 z 坐标的最小值和最大值
        self.xmin_Local = np.min(x_cart) if len(x_cart) > 1 else None
        self.xmax_Local = np.max(x_cart) if len(x_cart) > 1 else None
        self.ymin_Local = np.min(y_cart) if len(y_cart) > 1 else None
        self.ymax_Local = np.max(y_cart) if len(y_cart) > 1 else None
        self.zmin_Local = np.min(z_cart) if len(z_cart) > 1 else None
        self.zmax_Local = np.max(z_cart) if len(z_cart) > 1 else None

    def Get_SC_Grid_para_global(self):
        """
        使用MPI获取全局范围内的最小和最大x, y, z坐标。
        """
        # 先调用 Get_SC_Grid_para_Local 更新计算局部参数
        self.Get_SC_Grid_para_Local()

        comm = MPI.COMM_WORLD

        # 检查是否为有效值，避免NaN或None传递
        xmin_Local = self.xmin_Local if self.xmin_Local is not None else np.inf
        xmax_Local = self.xmax_Local if self.xmax_Local is not None else -np.inf
        ymin_Local = self.ymin_Local if self.ymin_Local is not None else np.inf
        ymax_Local = self.ymax_Local if self.ymax_Local is not None else -np.inf
        zmin_Local = self.zmin_Local if self.zmin_Local is not None else np.inf
        zmax_Local = self.zmax_Local if self.zmax_Local is not None else -np.inf

        # 通过allreduce获取全局的最小值和最大值
        self.xmin_Global = comm.allreduce(xmin_Local, op=MPI.MIN)
        self.xmax_Global = comm.allreduce(xmax_Local, op=MPI.MAX)
        self.ymin_Global = comm.allreduce(ymin_Local, op=MPI.MIN)
        self.ymax_Global = comm.allreduce(ymax_Local, op=MPI.MAX)
        self.zmin_Global = comm.allreduce(zmin_Local, op=MPI.MIN)
        self.zmax_Global = comm.allreduce(zmax_Local, op=MPI.MAX)

        # **扩大计算域范围**（例如扩展 20%）
        x_range = self.xmax_Global - self.xmin_Global
        y_range = self.ymax_Global - self.ymin_Global
        z_range = self.zmax_Global - self.zmin_Global

        self.xmin_Global -= 0.2 * x_range
        self.xmax_Global += 0.2 * x_range
        self.ymin_Global -= 0.2 * y_range
        self.ymax_Global += 0.2 * y_range
        self.zmin_Global -= 0.2 * z_range
        self.zmax_Global += 0.2 * z_range

        return self.xmin_Global, self.xmax_Global, self.ymin_Global, self.ymax_Global, self.zmin_Global, self.zmax_Global

    def DistributeChargeLocal(self, n, m, l):
        """
        根据 global 边界值和网格数量生成 3D 网格，并将 local bunch 中的粒子分配到 3D 网格中。
        :param n: x 方向的网格点数
        :param m: y 方向的网格点数
        :param l: z 方向的网格点数
        :return: local 3D 电荷分布矩阵
        """
        # 生成 3D 网格
        x_grid = np.linspace(self.xmin_Global, self.xmax_Global, n)
        y_grid = np.linspace(self.ymin_Global, self.ymax_Global, m)
        z_grid = np.linspace(self.zmin_Global, self.zmax_Global, l)
        Xmatrix, Ymatrix, Zmatrix = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        # 初始化电荷分布矩阵
        charge_distribution_local = np.zeros((n, m, l))

        # 获取所有本地粒子的笛卡尔坐标
        coordinates, _, _ = self.GetLocalCoordinates()

        # 使用 find_two_points_non_uniform_vect_njit 函数来找到粒子在网格中的邻近点
        idx_array_x, _ = find_two_points_non_uniform_vect_njit(x_grid, coordinates[:, 0])  # x 坐标
        idx_array_y, _ = find_two_points_non_uniform_vect_njit(y_grid, coordinates[:, 1])  # y 坐标
        idx_array_z, _ = find_two_points_non_uniform_vect_njit(z_grid, coordinates[:, 2])  # z 坐标

        # 调用 DistributeChargeNjit 来分配电荷到网格点
        charge_distribution_local_marco = DistributeChargeNjit(idx_array_x, idx_array_y, idx_array_z, x_grid, y_grid, z_grid,
                                                         charge_distribution_local, coordinates)

        charge_distribution_local_real = charge_distribution_local_marco * self.marcosize * FFAG_GlobalParameters().q

        # 保存本地的 3D 电荷分布矩阵
        self.charge_distribution_local = charge_distribution_local_real

        return charge_distribution_local_real, Xmatrix, Ymatrix, Zmatrix, x_grid, y_grid, z_grid

    def DistributeChargeGlobal(self, n, m, l):
        """
        汇总所有进程的局部电荷分布，得到全局的 3D 电荷分布矩阵。
        :param n: x 方向的网格点数
        :param m: y 方向的网格点数
        :param l: z 方向的网格点数
        :return: global 3D 电荷分布矩阵
        """
        # 初始化 MPI 通信
        comm = MPI.COMM_WORLD

        # 调用 DistributeChargeLocal 获取本地的 3D 电荷分布矩阵
        local_charge_distribution, Xmatrix, Ymatrix, Zmatrix, x_grid, y_grid, z_grid = (
            self.DistributeChargeLocal(n, m, l))

        # 初始化全局 3D 电荷分布矩阵
        global_charge_distribution = np.zeros((n, m, l))

        # 使用 MPI.allreduce 汇总所有进程的电荷分布矩阵
        comm.Allreduce(local_charge_distribution, global_charge_distribution, op=MPI.SUM)

        # 保存全局的 3D 电荷分布矩阵
        self.charge_distribution_global = global_charge_distribution
        self.Xmatrix = Xmatrix
        self.Ymatrix = Ymatrix
        self.Zmatrix = Zmatrix
        self.Xgrid = x_grid
        self.Ygrid = y_grid
        self.Zgrid = z_grid

        return global_charge_distribution, Xmatrix, Ymatrix, Zmatrix, x_grid, y_grid, z_grid

    def Update_SC_Efield_Local(self, Er_Local, Efi_Local, Ez_Local, Local_ID):
        """
        更新LocalBunch中的空间电荷效应self induced电场分量，包括径向电场Er，方位角电场Efi和纵向电场Ez。
        :param Er_Local: 空间电荷效应径向电场
        :param Efi_Local: 空间电荷效应方位角电场
        :param Ez_Local: 空间电荷效应纵向电场
        """
        # 获取电场对应的列索引
        er_idx = self.BunchAttribute.Attribute['Esc_r']
        ez_idx = self.BunchAttribute.Attribute['Esc_z']
        efi_idx = self.BunchAttribute.Attribute['Esc_fi']

        # 检查输入数组的长度是否匹配LocalBunch的粒子数
        if len(Er_Local) != len(Local_ID) or len(Efi_Local) != len(Local_ID) or len(Ez_Local) != len(
                Local_ID):
            raise ValueError("Input field arrays must have the same length as LocalBunch.")

        # 更新电场到LocalBunch的对应列
        self.LocalBunch[Local_ID, er_idx] = Er_Local
        self.LocalBunch[Local_ID, ez_idx] = Ez_Local
        self.LocalBunch[Local_ID, efi_idx] = Efi_Local


if __name__ == "__main__":
    # 初始化 MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 定义一个模拟的粒子分布
    np.random.seed(int(MPI.Wtime() * 1000) + rank)  # 使用当前时间和rank作为随机种子
    ParticleNum = 20
    # 创建包含 (r, vr, z, vz, fi, Ek, inj_t, ini_flag, RF_phase, Esc_r, Esc_z, Esc_fi, Bunch_ID, Local_ID, Global_ID) 的粒子
    ParticlesDistribution = np.column_stack((
        np.random.random(ParticleNum) * 10,  # r
        np.random.random(ParticleNum),  # vr
        np.random.random(ParticleNum) * 10,  # z
        np.random.random(ParticleNum),  # vz
        np.random.random(ParticleNum) * 2 * np.pi,  # fi
        np.random.random(ParticleNum) * 100,  # Ek
        np.random.random(ParticleNum) * 5,  # inj_t
        np.ones(ParticleNum),  # ini_flag
        np.random.random(ParticleNum) * 2 * np.pi,  # RF_phase
        np.zeros(ParticleNum),  # Esc_r (initially zero)
        np.zeros(ParticleNum),  # Esc_z (initially zero)
        np.zeros(ParticleNum),  # Esc_fi (initially zero)
        np.zeros(ParticleNum),  # Bunch_ID
        np.zeros(ParticleNum),  # Local_ID (will be set by the class)
        np.zeros(ParticleNum)  # Global_ID (will be set by the class)
    ))

    # 创建一个 FFAG_Bunch 实例
    bunch = FFAG_Bunch(ParticlesDistribution)

    # 在所有线程中计算全局的坐标边界
    xmin_global, xmax_global, ymin_global, ymax_global, zmin_global, zmax_global = bunch.Get_SC_Grid_para_global()

    # 每个线程打印其局部的和全局的坐标边界，保留两位小数
    print(f"Rank {rank}:")
    print(
        f"  Local xmin: {bunch.xmin_Local:.2f}, xmax: {bunch.xmax_Local:.2f}, ymin: {bunch.ymin_Local:.2f}, ymax: {bunch.ymax_Local:.2f}, zmin: {bunch.zmin_Local:.2f}, zmax: {bunch.zmax_Local:.2f}")
    print(
        f"  Global xmin: {xmin_global:.2f}, xmax: {xmax_global:.2f}, ymin: {ymin_global:.2f}, ymax: {ymax_global:.2f}, zmin: {zmin_global:.2f}, zmax: {zmax_global:.2f}\n")
