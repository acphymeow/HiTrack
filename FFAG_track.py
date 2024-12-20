"""
The Python-FFAG code is a RK solver for the beam dynamics simulations of the FFAG accelerator.
Contact Author: zhoukai@ihep.ac.cn
Version and Date:
V0.0 @ 2023.04.28
V0.1 @ 2023.11.20, Generating field maps, 6D tracking, search SEO, calculate twiss parameters
V0.2 @ 2024.11.01, 3d space charge modal, high-order tayler expansion modual
"""
import os
import math
import copy
import datetime
import time
import numpy as np
from mpi4py import MPI
from scipy.interpolate import interp1d
from FFAG_Utils import FFAG_FileOperation
from FFAG_MathTools import FFAG_interpolation, FFAG_Algorithm
from FFAG_ParasAndConversion import FFAG_ConversionTools
from FFAG_SC import Bunch_SC_Calculator


class FFAG_RungeKutta:
    def __init__(self):
        self.running = True

    def rk4_step(self, func, t, r, h, GlobalParameters):
        # shape of r = number of particles * number of coordinates
        k1 = h * func(t, r, GlobalParameters)
        k2 = h * func(t + 0.5 * h, r + 0.5 * k1, GlobalParameters)
        k3 = h * func(t + 0.5 * h, r + 0.5 * k2, GlobalParameters)
        k4 = h * func(t + h, r + k3, GlobalParameters)
        # shape of k1,...,k4 = n*7
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def rk4_step_2(self, func, t, r, h, GlobalParameters):
        k1 = h * func(t, r, GlobalParameters)[0]
        k2 = h * func(t + 0.5 * h, r + 0.5 * k1, GlobalParameters)[0]
        k3 = h * func(t + 0.5 * h, r + 0.5 * k2, GlobalParameters)[0]
        k4 = h * func(t + h, r + k3, GlobalParameters)[0]
        _, Bz_ThisStep, Br_ThisStep, Bfi_ThisStep = func(t, r, GlobalParameters)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6, Bz_ThisStep, Br_ThisStep, Bfi_ThisStep


    def rk4_solve_dt_bunch3_withSC(self, func, t_start, BunchObj,
                            GlobalParameters, LocalParameters):

        comm = MPI.COMM_WORLD

        # parameters for tracking
        stop_condition = LocalParameters['stop_condition']
        step_condition = LocalParameters['step_condition']
        enable_SC = LocalParameters['enable_SC']
        StepDumpInfo = LocalParameters['step_dumps']

        max_stepsN = stop_condition['max_stepsN']
        max_turn = stop_condition['max_turn']
        max_time = stop_condition['max_time']
        max_fi = max_turn * np.pi * 2
        step_fi = np.deg2rad(step_condition['step_fi'])
        max_step_fi = np.deg2rad(step_condition['max_step_fi'])
        min_step_fi = np.deg2rad(step_condition['min_step_fi'])

        # Set the length of initial time step
        # dfidt_mean = BunchObj.GetMeanDfiDt()
        # step_t = np.abs(step_fi / dfidt_mean) * TrackDirection
        step_t = 20e-11
        steps_oneturn = np.floor(np.pi * 2 / step_fi)

        # Set the initial Matrix
        InitParticleNum = BunchObj.TotalParticleNum

        t_PreStep, steps_i = t_start, 0
        ExitLoopFlag = False

        # start tracking
        while steps_i < max_stepsN:

            BunchObj.UpdateAndInjectParticles(t_PreStep)
            BunchObj.UpdatePreSteps()
            if ExitLoopFlag:
                # if there are no particles in the threads,
                # execute an empty statement to ensure synchronization among threads
                pass
            else:
                # the pre step
                r_PreStep = copy.deepcopy(BunchObj.LocalBunch)

                if enable_SC:
                    if np.mod(steps_i, 8) == 0:
                        # Step 3: 计算空间电荷电场
                        BunchObj.UpdateGlobalBunch()
                        BunchObj.Get_SC_Grid_para_global()
                        BunchObj.DistributeChargeGlobal(32, 32, 32)

                        # Step 4: 计算空间电荷电场, 将插值后的电场值存储到 Bunch_obj中
                        _, _, _, _, _, _, _, _ = (
                            Bunch_SC_Calculator(BunchObj))

                # the integration over the step
                dr_step, Bz_ThisStep, Br_ThisStep, Bfi_ThisStep = self.rk4_step_2(func, t_PreStep, r_PreStep, step_t, GlobalParameters)

                # the post step
                BunchObj.LocalBunch = BunchObj.LocalBunch + dr_step * np.tile(r_PreStep[:, 7], (15, 1)).T
                t_PostStep = t_PreStep + step_t

                # 更新 Post-steps (积分之后)
                BunchObj.UpdatePostSteps()

                #  save coordinates with the given time interval
                StepDumpInfo.check_and_dump(steps_i, t_PostStep, BunchObj)

                # update the pre step
                t_PreStep = t_PostStep
                if np.mod(steps_i, 1000) == 0:
                    if comm.Get_rank() == 0:
                        print(
                            f"current step={steps_i:.0f}, fi = {np.min(r_PreStep[:, 4]):.2f} rad, turn = {np.min(r_PreStep[:, 4]) / np.pi / 2:.2f}. current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)

                # # # variable time steps
                # # check the step length every 1/2 turn, adapt the step length if needed
                # if np.mod(steps_i, int(steps_oneturn / 2)) == 0:
                #     dfidt_mean = BunchObj.GetMeanDfiDt()
                #     step_fi_ThisStep = dfidt_mean * step_t
                #     if np.abs(step_fi_ThisStep) > np.abs(max_step_fi) or np.abs(step_fi_ThisStep) < np.abs(min_step_fi):
                #         step_t = step_fi / dfidt_mean * TrackDirection

                # update loop index
                steps_i += 1

            comm.barrier()

            # Exit the loop
            if np.mod(steps_i, 1000) == 0:
                # stop conditions
                StopFlag_fi_local = copy.deepcopy(BunchObj.LocalBunch[:, 4])
                StopFlag_time_local = t_PreStep
                # 使用allgather将各线程汇总到一个进程中
                StopFlag_fi_gather = comm.allgather(StopFlag_fi_local)
                StopFlag_time_gather = comm.allgather(StopFlag_time_local)
                # 根进程拥有拼接后的数据
                StopFlag_fi_global = np.concatenate(StopFlag_fi_gather, axis=0)
                StopFlag_time_global = np.mean(StopFlag_time_gather)
                StopFlag_fi_survived_particles = StopFlag_fi_global[~np.isnan(StopFlag_fi_global)]

                if np.count_nonzero(np.isnan(StopFlag_fi_global)) > 0:
                    pass

                if np.abs(np.mean(StopFlag_fi_survived_particles)) > max_fi:
                    break
                if StopFlag_time_global > max_time:
                    break

        return 0

    def rk4_solve_vect(self, func, t_start, t_end, r_start, h, GlobalParameters, LocalParameters=False):
        steps = int((t_end - t_start) / h)
        t_points = np.linspace(t_start, t_end, steps)  # independent variables
        steps_real = t_points[1] - t_points[0]
        NSteps = len(t_points)
        t_points_AllSteps, r_points_AllSteps, rt_points_AllSteps = [], [], []
        r_points = copy.deepcopy(r_start)

        t_PreStep, i = t_points[0], 0

        while i < NSteps:
            # the pre step
            r_PreStep = copy.deepcopy(r_points)
            tr_PreStep = np.column_stack((np.ones_like(r_PreStep[:, 0]) * t_PreStep, r_PreStep))

            t_points_AllSteps.append(t_PreStep)
            r_points_AllSteps.append(r_PreStep)
            rt_points_AllSteps.append(tr_PreStep)

            dr_step = self.rk4_step(func, t_PreStep, r_PreStep, steps_real, GlobalParameters)
            r_PostStep = r_PreStep + dr_step
            t_PostStep = t_PreStep + steps_real

            # update loop index
            i += 1

            # update the pre step
            r_points = copy.deepcopy(r_PostStep)
            t_PreStep = t_PostStep

            # beam lost condition
            rstop_max = GlobalParameters.Bmap.r_max + GlobalParameters.Bmap.r_step * 10
            rstop_min = GlobalParameters.Bmap.r_min - GlobalParameters.Bmap.r_step * 10
            flag_delete_condition = (r_points[:, 0] < rstop_min) | (r_points[:, 0] > rstop_max)
            r_points = np.delete(r_points, flag_delete_condition, axis=0)

        return rt_points_AllSteps[-1]

    def rk4_solve(self, func, t_start, t_end, r_start, h, GlobalParameters):
        steps = int((t_end - t_start) / h)
        t_points = np.linspace(t_start, t_end, steps)  # independent variables
        steps_real = t_points[1] - t_points[0]
        t_length, r_length = len(t_points), len(r_start)  # steps
        r_points = np.zeros((t_length, r_length))  # coordinates for all steps
        r_points[0, :] = r_start  # set the first step

        for i in range(len(t_points) - 1):
            r_PreStep = r_points[i, :].copy()  # the pre step
            t_PreStep = t_points[i]

            # the post step
            dr_step = self.rk4_step(func, t_PreStep, r_PreStep, steps_real, GlobalParameters)
            r_PostStep = r_PreStep + dr_step

            r_points[i + 1, :] = r_PostStep
        return t_points, r_points


    def rk4_solve_vect_3DMatrix(self, func, t_start, t_end, r_start, h, GlobalParameters, LocalParameters=False):

        steps = int((t_end - t_start) / h)
        t_points = np.linspace(t_start, t_end, steps)  # independent variables

        steps_real = t_points[1] - t_points[0]
        NSteps = len(t_points)
        # t_points_AllSteps, r_points_AllSteps, rt_points_AllSteps = [], [], []
        r_points = copy.deepcopy(r_start)
        InitParticleNum = np.size(r_points, 0)
        # Set the initial matrix
        tr_points_AllSteps = np.zeros((0, InitParticleNum, np.size(r_points, 1) + 1))

        t_PreStep, i = t_points[0], 0

        while i < NSteps:
            # the pre step
            r_PreStep = copy.deepcopy(r_points)
            tr_points_ThisStep = np.column_stack((np.ones_like(r_PreStep[:, 0]) * t_PreStep, r_PreStep))
            tr_points_AllSteps = np.concatenate((tr_points_AllSteps, tr_points_ThisStep[np.newaxis, :, :]), axis=0)

            dr_step = self.rk4_step(func, t_PreStep, r_PreStep, steps_real, GlobalParameters)

            r_PostStep = r_PreStep + dr_step
            t_PostStep = t_PreStep + steps_real

            # update loop index
            i += 1

            # update the pre step
            r_points = copy.deepcopy(r_PostStep)
            t_PreStep = t_PostStep

            # beam lost condition
            rstop_max = GlobalParameters.Bmap.r_max + GlobalParameters.Bmap.r_step * 10
            rstop_min = GlobalParameters.Bmap.r_min - GlobalParameters.Bmap.r_step * 10
            flag_delete_condition = (r_points[:, 0] < rstop_min) | (r_points[:, 0] > rstop_max)
            r_points = np.delete(r_points, flag_delete_condition, axis=0)

        return tr_points_AllSteps[-1, :, :], tr_points_AllSteps


class FFAG_MPI:
    def __init__(self):
        pass

    def DivideVariables(self, xGlobal):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        TotalNum = len(xGlobal)
        MinNumPerCPU = math.floor(TotalNum / size)
        ModNum = TotalNum - MinNumPerCPU * size
        xLocalIndex = list(range(rank * MinNumPerCPU, (rank + 1) * MinNumPerCPU))
        if rank < ModNum:
            xLocalIndex.append(MinNumPerCPU * size + rank)
        xLocal = [xGlobal[i] for i in xLocalIndex]
        # print(f"Rank {MPI.COMM_WORLD.Get_rank()}: xLocal = {xLocal}")
        return np.array(xLocal, dtype=int)


class FFAG_SearchSEO:
    def __init__(self, GlobalParas):
        self.GlobalParas = GlobalParas

    def ObjectiveFuncForSEO(self, x, ObjFuncPara):
        Ek_value = ObjFuncPara.Ek_value
        t_end = ObjFuncPara.t_end
        h = ObjFuncPara.h
        r0, pr0 = x[0], x[1]
        t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, t_end, h)
        ee = ((r_points[0, 0] - r_points[-1, 0]) / 1e-6) ** 2 + ((r_points[0, 1] - r_points[-1, 1]) / 1e-6) ** 2
        return ee

    def TrackAParticle(self, r0, pr0, Ek_value, t_end, h):

        rk = FFAG_RungeKutta()
        t_start = 0
        P_start = FFAG_ConversionTools().Ek2P(Ek_value)
        ini_no_offset = np.array([r0, pr0, 0, 0, P_start, 0])  # r,pr,z,pz,P,index
        # if SearchingMod:
        tr_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_no_offset, h,
                                          self.GlobalParas)
        return tr_points


    def TrackBunchSEO(self, r0, p0, Ek_value, t_end, h, verbose):
        t_start = 0
        rk = FFAG_RungeKutta()

        P_start = FFAG_ConversionTools().Ek2P(Ek_value)
        # r,pr,z,pz,P,index
        delta_r, delta_p = r0 * 0.0001, 0.001
        for k in range(100):
            Ini_start = np.array([[r0, p0, 0, 0, P_start, 0],
                                  [r0 + delta_r, p0, 0, 0, P_start, 1],
                                  [r0, p0 + delta_p, 0, 0, P_start, 2]])

            tr_points = rk.rk4_solve_vect(FunctionForSEOVect, t_start, t_end, Ini_start, h, self.GlobalParas)
            r0f, p0f = tr_points[0, 1], tr_points[0, 2]
            r1f, p1f = tr_points[1, 1], tr_points[1, 2]
            r2f, p2f = tr_points[2, 1], tr_points[2, 2]

            a11 = (r1f - r0f) / delta_r
            a12 = (r2f - r0f) / delta_p
            a21 = (p1f - p0f) / delta_r
            a22 = (p2f - p0f) / delta_p
            a11_prime, a22_prime = a11 - 1, a22 - 1
            DetermineA = a11_prime * a22_prime - a12 * a21

            if verbose:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('In accurate calculation, (r0f - r0)/r0 = ', (r0f - r0) / r0)
            if np.abs((r0f - r0) / r0) < 1e-8:
                break

            re = r0 + a22_prime / DetermineA * (r0 - r0f) - a12 / DetermineA * (p0 - p0f)
            pe = p0 + a11_prime / DetermineA * (p0 - p0f) - a21 / DetermineA * (r0 - r0f)

            r0 = re
            p0 = pe

        return r0, p0


    # def SearchSEOUsingInitialEkVect(self, Ek_value, h=0.001, delta_r0_ini=0.0, verbose=False):
    #     """
    #     search for the SEO for a given Ek value
    #     return: initial condition r0, pr0
    #     """
    #     self.GlobalParas.TempVariable = Ek_value
    #     # Find analytical initial value r0, pr0
    #     r0_analytical, _ = FFAG_Algorithm().BiSection(self.GlobalParas.Bmap.r_min, self.GlobalParas.Bmap.r_max,
    #                                                   func_get_R_B0, self.GlobalParas)
    #     r0 = r0_analytical + delta_r0_ini
    #     pr0 = 0.0
    #
    #     t_end = np.pi * 2 / self.GlobalParas.Bmap.Nsectors
    #     # print("Nsectors=", self.GlobalParas.Bmap.Nsectors)
    #     # Find approximate initial value r0, pr0   30
    #     # for i_nSector in np.arange(0, self.GlobalParas.Bmap.Nsectors):
    #     for i_nSector in [0, ]:
    #         t_end_i = t_end * (i_nSector + 1)
    #
    #         for k in np.arange(0, 100):
    #             t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, t_end_i, h)
    #
    #             r_of_fi, pr_of_fi = r_points[:, 0], r_points[:, 1]
    #             r0, pr0 = (r_of_fi[0] + r_of_fi[-1]) / 2, (pr_of_fi[0] + pr_of_fi[-1]) / 2
    #             if verbose:
    #                 if MPI.COMM_WORLD.Get_rank() == 0:
    #                     print('i=', Ek_value, ' k=', k, 'ee=',
    #                           np.sqrt(
    #                               ((r_of_fi[0] - r_of_fi[-1]) / 1e-6) ** 2 + ((pr_of_fi[0] - pr_of_fi[-1]) / 1e-6) ** 2),
    #                           'r_start=', r_of_fi[0], 'r_end=', r_of_fi[-1])
    #
    #             if np.sqrt(((r_of_fi[0] - r_of_fi[-1]) / 1e-6) ** 2 + ((pr_of_fi[0] - pr_of_fi[-1]) / 1e-6) ** 2) < 1:
    #                 break
    #
    #     r0, pr0 = self.TrackBunchSEO(r0, pr0, Ek_value, t_end, h, verbose)
    #
    #     return r0, pr0, r0_analytical

    def SearchSEOUsingInitialEkVect(self, Ek_value, h=0.001, delta_r0_ini=0.0, verbose=False):
        """
        search for the SEO for a given Ek value
        return: initial condition r0, pr0
        """
        self.GlobalParas.TempVariable = Ek_value
        # Find analytical initial value r0, pr0
        r0_analytical, _ = FFAG_Algorithm().BiSection(self.GlobalParas.Bmap.r_min, self.GlobalParas.Bmap.r_max,
                                                      func_get_R_B0, self.GlobalParas)
        r0 = r0_analytical + delta_r0_ini

        t_end_i = np.pi * 2 / self.GlobalParas.Bmap.Nsectors

        # 定义调整步长的列表
        pr0_list = [0.00, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06]

        # 标志：表示是否已经找到符合条件的 r0 和 pr0
        found = False

        # Find approximate initial value r0, pr0
        for pr0 in pr0_list:

            # 重置初始值 r0 和 pr0
            r0 = r0_analytical + delta_r0_ini

            for k in np.arange(0, 100):
                # 跟踪粒子轨迹
                t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, t_end_i, h)

                # 提取位置和动量
                r_of_fi, pr_of_fi = r_points[:, 0], r_points[:, 1]

                # 更新初始值 r0 和 pr0
                r0, pr0 = (r_of_fi[0] + r_of_fi[-1]) / 2, (pr_of_fi[0] + pr_of_fi[-1]) / 2

                # 打印调试信息
                if verbose:
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print('i=', Ek_value, ' k=', k, 'ee=',
                              np.sqrt(
                                  ((r_of_fi[0] - r_of_fi[-1]) / 1e-6) ** 2 + (
                                          (pr_of_fi[0] - pr_of_fi[-1]) / 1e-6) ** 2),
                              'r_start=', r_of_fi[0], 'r_end=', r_of_fi[-1])

                # 判断是否出现NaN情况
                if np.isnan(r_of_fi[-1]) or np.isnan(pr_of_fi[-1]):
                    # 如果是NaN，退出当前 100 次迭代
                    print(f"NaN detected at iteration {k}. Adjusting pr0 and retrying.")

                    # 恢复到最初的 r0 和 delta_r0_ini
                    r0 = r0_analytical + delta_r0_ini

                    # 跳过当前的 100 次循环，继续尝试下一个 pr0
                    break  # 跳出当前 100 次迭代，继续尝试下一个 pr0

                # 终止条件：当位置和动量的变化量小于某个阈值时跳出循环
                if np.abs(r_of_fi[0] - r_of_fi[-1]) < 1e-6 and np.abs(pr_of_fi[0] - pr_of_fi[-1]) < 1e-6:
                    found = True  # 设置标志为 True，表示满足条件
                    break  # 跳出当前 100 次迭代

            if found:
                # 如果满足条件，跳出 pr0_list 循环
                break  # 跳出 pr0_list 循环

        # 终止后的结果处理
        if found:
            r0, pr0 = self.TrackBunchSEO(r0, pr0, Ek_value, t_end_i, h, verbose)
        else:
            raise RuntimeError(f"No close orbit solution found for {Ek_value}MeV.")

        return r0, pr0, r0_analytical

    def GetQrQz(self, r0, pr0, Ek_value, h):
        t_start = 0
        t_end = np.pi * 2 / self.GlobalParas.Bmap.Nsectors
        z0, pz0 = 0.0, 0.0
        delta_rr, delta_pp = 1e-10, 1e-10
        delta_zz, delta_pz = 1e-8, 1e-8

        P_start = FFAG_ConversionTools().Ek2P(Ek_value)
        ini_no_offset = np.array([r0, pr0, z0, pz0, P_start, 0])
        ini_offset_R = np.array([r0 + delta_rr, pr0, z0, pz0, P_start, 0])
        ini_offset_Pr = np.array([r0, pr0 + delta_pp, z0, pz0, P_start, 0])
        ini_offset_Z = np.array([r0, pr0, z0 + delta_zz, pz0, P_start, 0])
        ini_offset_PZ = np.array([r0, pr0, z0, pz0 + delta_pz, P_start, 0])
        rk = FFAG_RungeKutta()
        # r1, pr1, z1, pz1
        t_points, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_no_offset, h,
                                          self.GlobalParas)
        r1, p1, z1, pz1 = r_points[-1, 0], r_points[-1, 1], 0.0, 0.0
        OrbitFreq = self.GetFreq(t_points, r_points[:, 0], r_points[:, 1], Ek_value)
        Mean_R = np.mean(r_points[:, 0])

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_R, h,
                                   self.GlobalParas)
        r2, p2 = r_points[-1, 0], r_points[-1, 1]

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_Pr, h,
                                   self.GlobalParas)
        r3, p3 = r_points[-1, 0], r_points[-1, 1]

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_Z, h,
                                   self.GlobalParas)
        z2, pz2 = r_points[-1, 2], r_points[-1, 3]

        _, r_points = rk.rk4_solve(FunctionForSEO, t_start, t_end, ini_offset_PZ, h,
                                   self.GlobalParas)
        z3, pz3 = r_points[-1, 2], r_points[-1, 3]

        a11 = (r2 - r1) / delta_rr
        a22 = (p3 - p1) / delta_pp
        b11 = (z2 - z1) / delta_zz
        b22 = (pz3 - pz1) / delta_pz
        print(f"b11={b11}, b22={b22}")
        Qr_float2 = np.arccos((a11 + a22) / 2) / (2 * np.pi) * self.GlobalParas.Bmap.Nsectors
        Qz_float2 = np.arccos((b11 + b22) / 2) / (2 * np.pi) * self.GlobalParas.Bmap.Nsectors
        # Qr_float2 = np.arccos((a11 + a22) / 2) / (2 * np.pi)
        # Qz_float2 = np.arccos((b11 + b22) / 2) / (2 * np.pi)
        return Qr_float2, Qz_float2, OrbitFreq, Mean_R

    def GetFreq(self, fi_SEO, r_SEO, pr_SEO, Ek_value):
        """
        Return the orbital frequency for the given Ek value
        """
        rp_SEO = pr_SEO / (np.sqrt(1 - pr_SEO ** 2)) * r_SEO
        DiffArcLength = np.sqrt(r_SEO ** 2 + rp_SEO ** 2)
        DiffArcLengthDfi = DiffArcLength * (fi_SEO[2] - fi_SEO[1])
        Perimeter = sum(DiffArcLengthDfi[:-1]) * self.GlobalParas.Bmap.Nsectors
        v = FFAG_ConversionTools().Ek2v(Ek_value)
        Period = Perimeter / v
        frequency = 1 / Period
        return frequency

    def SearchSEOsControllerVect(self, EkRange, SavedataFlag=True):
        EkIndex = np.arange(0, len(EkRange))
        EkIndexLocal = FFAG_MPI().DivideVariables(EkIndex)
        EkNumLocal, EkNumTotal = len(EkIndexLocal), len(EkIndex)

        SEOdataLocal = np.zeros((EkNumLocal, 16))  # Ek_index, Ek_value, r0, pr0, r_end, pr_end, Qr, Qz, freq, MeanR
        SEO_fir_Local = []
        SEO_fipr_Local = []
        SEO_fi_Local = []
        h = 0.0002

        index = 0
        delta_r0_ini = 0.0
        for Ek_index in EkIndexLocal:
            Ek_value = EkRange[Ek_index]
            r0, pr0, r0_analytical = self.SearchSEOUsingInitialEkVect(Ek_value, h,
                                                                      delta_r0_ini, verbose=False)  # the accurate initial value
            t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, np.pi * 2, 0.001)
            r_of_fi, pr_of_fi = r_points[:, 0], r_points[:, 1]

            Ek_fi_label = np.hstack((Ek_value, t_points))
            Ek_r_of_fi = np.hstack((Ek_value, r_of_fi))
            Ek_pr_of_fi = np.hstack((Ek_value, pr_of_fi))

            SEO_fi_Local.append(Ek_fi_label)
            SEO_fipr_Local.append(Ek_pr_of_fi)
            SEO_fir_Local.append(Ek_r_of_fi)

            # 替换线性插值器，改用 SciPy 的 interp1d
            rfi_interp = interp1d(t_points, r_of_fi, kind='linear', fill_value="extrapolate")
            prfi_interp = interp1d(t_points, pr_of_fi, kind='linear', fill_value="extrapolate")

            # 使用插值器计算特定角度下的 r 和 pr 值
            r90, pr90 = rfi_interp(np.deg2rad(90.0)), prfi_interp(np.deg2rad(90.0))
            r180, pr180 = rfi_interp(np.deg2rad(180.0)), prfi_interp(np.deg2rad(180.0))
            r270, pr270 = rfi_interp(np.deg2rad(270.0)), prfi_interp(np.deg2rad(270.0))

            Qr, Qz, OrbitFreq, MeanR = self.GetQrQz(r0, pr0, Ek_value, h)
            print(f"Closed orbit found for {Ek_value}MeV, Qr={Qr:.3f}, Qz={Qz:.3f}")

            SEOdataLocal[index, :] = np.array(
                (Ek_index, Ek_value, Qr, Qz, OrbitFreq, MeanR,
                 r0, pr0, r_points[-1, 0], r_points[-1, 1],
                 r90, pr90, r180, pr180, r270, pr270,))

            index += 1

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        SEOdataGlobal = comm.allgather(SEOdataLocal)
        SEO_fir_Global = comm.allgather(SEO_fir_Local)
        SEO_fipr_Global = comm.allgather(SEO_fipr_Local)
        SEO_fi_Global = comm.allgather(SEO_fi_Local)

        SEO_foldname = None
        # 根进程拥有拼接后的数据
        if rank == 0:
            SEOdataGlobal_arr = np.concatenate(SEOdataGlobal, axis=0)

            SEO_fir_Global_arr = np.concatenate(SEO_fir_Global, axis=0)
            SEO_fipr_Global_arr = np.concatenate(SEO_fipr_Global, axis=0)
            SEO_fi_Global_arr = np.concatenate(SEO_fi_Global, axis=0)
            # 获取按第一列排序的索引
            sorted_indices = np.argsort(SEOdataGlobal_arr[:, 0])
            sorted_indices_rpr = np.argsort(SEO_fir_Global_arr[:, 0])
            # 使用索引对数组进行排序
            SEOdataGlobal_arr_sorted = SEOdataGlobal_arr[sorted_indices]

            SEO_fir_Global_arr_sorted = SEO_fir_Global_arr[sorted_indices_rpr]
            SEO_fipr_Global_arr_sorted = SEO_fipr_Global_arr[sorted_indices_rpr]
            SEO_fi_Global_arr_sorted = SEO_fi_Global_arr[sorted_indices_rpr]

            ExcelTitles = ["Ek_index", "Ek_value(MeV)", "Qr", "Qz", "OrbitFreq(Hz)", "MeanR(m)",
                           "r0(m)", "pr0", "r360(m)", "pr360",
                           "r90(m)", "pr90", "r180(m)", "pr180", "r270(m)", "pr270"]

            if SavedataFlag:
                now = datetime.datetime.now()
                BmapFoldname = os.path.splitext(os.path.basename(self.GlobalParas.Bmap.BmapFileName))[0]
                SEO_foldname = './resultsSEO/' + BmapFoldname + '-%d-%02d-%02d-%02d%02d%02d' % (
                    now.year, now.month, now.day, now.hour, now.minute, now.second)
                FFAG_FileOperation().CreatAEmptyFold_safe(SEO_foldname)
                print(f"SEO information will be writen in folder '{SEO_foldname}'.")

                SEO_foldfilename = SEO_foldname + '/SEO_ini.txt'
                SEO_fipr_filename = SEO_foldname + '/SEO_pr.txt'
                SEO_fir_filename = SEO_foldname + '/SEO_r.txt'

                # 合并表头和数据矩阵为一个整体的列表
                SEOdataGlobal_arr_sorted_float = SEOdataGlobal_arr_sorted.astype(float)
                SEOdataGlobal_arr_sorted_float[:, 2:6] = np.round(SEOdataGlobal_arr_sorted_float[:, 2:6], 3)
                SEOdataGlobal_arr_sorted_float[:, 6:] = np.round(SEOdataGlobal_arr_sorted_float[:, 6:], 8)

                combined_data = [ExcelTitles] + SEOdataGlobal_arr_sorted_float.tolist()

                # 计算每列的最大宽度，包括表头的每列
                max_widths = [max(len(str(row[i])) for row in combined_data) for i in range(len(combined_data[0]))]

                # 格式化并左对齐每列，包括表头，并在相邻两列之间插入两列空格
                formatted_data = []
                for row in combined_data:
                    formatted_row = ["  ".join([str(value).ljust(width) for value, width in zip(row, max_widths)])]
                    formatted_data.extend(formatted_row)

                # 在首行添加Bmap相关信息
                TheFirstLine = "Bmap filename=" + self.GlobalParas.Bmap.BmapFileName + ", step h = %.6f rad" % (h,)
                formatted_data.insert(0, TheFirstLine)

                # 将格式化后的数据写入文本文件
                with open(SEO_foldfilename, "w") as file:
                    file.write("\n".join(formatted_data))

                SEO_fir_Global_matrix = np.row_stack((SEO_fi_Global_arr_sorted[0, :], SEO_fir_Global_arr_sorted))
                SEO_fipr_Global_matrix = np.row_stack((SEO_fi_Global_arr_sorted[0, :], SEO_fipr_Global_arr_sorted))

                with open(SEO_fir_filename, "w") as fid_fir:
                    np.savetxt(fid_fir, SEO_fir_Global_matrix)

                with open(SEO_fipr_filename, "w") as fid_fipr:
                    np.savetxt(fid_fipr, SEO_fipr_Global_matrix)

        return SEO_foldname

    def task(self, Ek_value):
        h = 0.001
        delta_r0_ini = -0.05
        r0, pr0, r0_analytical = self.SearchSEOUsingInitialEkVect(Ek_value, h,
                                                                  delta_r0_ini, verbose=False)  # the accurate initial value
        t_points, r_points = self.TrackAParticle(r0, pr0, Ek_value, np.pi * 2 / self.GlobalParas.Bmap.Nsectors, h)
        r_of_fi, pr_of_fi = r_points[:, 0], r_points[:, 1]

        orbitalfreq = self.GetFreq(t_points, r_of_fi, pr_of_fi, Ek_value)

        return orbitalfreq


class FFAG_DynamicAperture:
    def __init__(self):
        pass

    def SearchDAUsingInitialEkVect(self):
        pass


def FunctionForSEO(t, x, GlobalParameters):
    q = GlobalParameters.q
    dxdt = np.zeros(6)
    Bmap = GlobalParameters.Bmap
    NSectors=Bmap.Nsectors
    fi = t % (np.pi * 2)
    r, pr, z, pz, P, ParticleIndex = x[0], x[1], x[ 2], x[3], x[ 4], x[ 5]

    r_vect = np.array([r,])
    fi_vect = np.array([fi,])
    fi_vect_mod = np.mod(fi_vect, 2*np.pi/NSectors)

    if not Bmap.flag3D:
        Bz0_vect, _ = Bmap.Interpolation2DMap(r_vect, fi_vect_mod, 0, 3)  # T
        Bz0 = Bz0_vect[0]
        Bfi = Bz0*0
        Br = Bz0*0
    else:
        Bz0_vect, _ = Bmap.Interpolation2DMap(r_vect, fi_vect_mod, 0,  0) # T
        patial_r_vect, _ = Bmap.Interpolation2DMap(r_vect, fi_vect_mod, 0,  2)  # T / m
        patial_theta_vect, _ = Bmap.Interpolation2DMap(r_vect, fi_vect_mod, 0,  1)  # T / rad
        Bz0 = Bz0_vect[0]
        Bfi = patial_theta_vect[0] * z
        Br = patial_r_vect[0] * z

    pfi = np.sqrt((1 - pr ** 2 - pz ** 2))

    dxdt[0] = r * pr / pfi
    dxdt[1] = pfi - q / P * (r * Bz0 - r * pz / pfi * Bfi)
    dxdt[2] = r * pz / pfi
    dxdt[3] = q / P * (r * Br - r * pr / pfi * Bfi)
    dxdt[4] = 0
    dxdt[5] = 0

    return dxdt


def FunctionForSEOVect(t, x, GlobalParameters):
    q = GlobalParameters.q
    nParticles = np.size(x, 0)  # 获取粒子数量
    dxdt = np.zeros((nParticles, 6))
    Bmap = GlobalParameters.Bmap
    NSectors = Bmap.Nsectors
    max_order = Bmap.max_order
    fi = np.ones(nParticles) * (t % (np.pi * 2))  # 为每个粒子生成相同的时间角度

    # 提取粒子状态
    r, pr, z, pz, P, ParticleIndex = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]

    # 调整角度以适应扇区数
    fi_vect_mod = np.mod(fi, 2 * np.pi / NSectors)

    # 初始化 Bz、Br 和 Bfi 分量
    Bz0 = np.zeros_like(r)
    Br = np.zeros_like(r)
    Bfi = np.zeros_like(r)

    if not Bmap.flag3D:
        Bz_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect_mod, 0, 0)
        # 只有Bz
        Bz0 += Bz_coeff
    else:
        # 动态添加高阶项，根据 max_order 自动展开
        for n in range(0, max_order + 1):
            # 获取每阶的拉普拉斯项
            Bz_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect_mod, n, 0)
            # Bz 只有偶次项 (z^(2n))
            Bz0 += (z ** (2 * n)) * Bz_coeff

            # Br 和 Bfi 只有奇次项 (z^(2n+1))
            Br_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect_mod, n, 2)
            Bfi_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect_mod, n, 1)
            Br += (z ** (2 * n + 1)) * Br_coeff
            Bfi += (z ** (2 * n + 1)) * Bfi_coeff

    # 动量分量 pfi
    pfi = np.sqrt(1 - pr ** 2 - pz ** 2)

    # 计算 dxdt 各分量
    dxdt[:, 0] = r * pr / pfi
    dxdt[:, 1] = pfi - q / P * (r * Bz0 - r * pz / pfi * Bfi)
    dxdt[:, 2] = r * pz / pfi
    dxdt[:, 3] = q / P * (r * Br - r * pr / pfi * Bfi)
    dxdt[:, 4] = 0
    dxdt[:, 5] = 0

    return dxdt


def FunctionForAccelerationBunch_dt(t, x, GlobalParameters):
    # shape of x: number of particles * number of coordinates in formula
    # sequence of the columns: (r, vr), (z, vz), (fi, dfidt), (t_inj, inj_flag),
    # (rf_phase, Esc_r, Esc_z, Esc_fi), (Bunch_ID, Local_ID, Global_ID)

    (r, rdot, z, zdot, fi, Etotal_J, t_inj, flag_inj,
     RF_Phase, Esc_r, Esc_z, Esc_fi, BID_local, PID_local, PID_global) = \
        (x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5], x[:, 6],
         x[:, 7], x[:, 8], x[:, 9], x[:, 10], x[:, 11], x[:, 12], x[:, 13], x[:, 14])

    nParticles = x.shape[0]
    q = GlobalParameters.q
    c = GlobalParameters.c
    E0_J = GlobalParameters.E0
    Bmap = GlobalParameters.Bmap
    NSectors = Bmap.Nsectors
    max_order = Bmap.max_order

    gamma = Etotal_J / E0_J
    beta = np.sqrt(1 - 1 / (gamma ** 2))
    v = beta * c
    fidot = np.sqrt(v ** 2 - rdot ** 2 - zdot ** 2) / r
    fi_vect = fi % (np.pi * 2 / NSectors)

    # 初始化 Bz、Br 和 Bfi 分量
    Bz0 = np.zeros_like(r)
    Br = np.zeros_like(r)
    Bfi = np.zeros_like(r)

    # 动态添加高阶项，根据 max_order 自动展开
    for n in range(0, max_order + 1):
        # 获取每阶的拉普拉斯项
        Bz_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect, n, 0)
        Bz0 += (z ** (2 * n)) * Bz_coeff  # Bz 只包含偶次项

        Br_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect, n, 2)
        Bfi_coeff, _ = Bmap.Interpolation2DMap(r, fi_vect, n, 1)
        Br += (z ** (2 * n + 1)) * Br_coeff  # Br 只包含奇次项
        Bfi += (z ** (2 * n + 1)) * Bfi_coeff  # Bfi 只包含奇次项

    # 电场分量设为 0
    E_z, E_r, E_fi = 0.0, 0.0, 0.0

    # 考虑空间电荷的电场分量
    Ez_tot = E_z + Esc_z
    Er_tot = E_r + Esc_r
    Efi_tot = E_fi + Esc_fi

    # 初始化 dxdt 数组并计算状态导数
    dxdt = np.zeros((nParticles, 15))
    dxdt[:, 0] = rdot
    dxdt[:, 1] = (r * fidot ** 2 + q * c ** 2 / Etotal_J * (Er_tot - r * fidot * Bz0 + zdot * Bfi)
                  - q * rdot / Etotal_J * (rdot * Er_tot + r * fidot * Efi_tot + zdot * Ez_tot))
    dxdt[:, 2] = zdot
    dxdt[:, 3] = (q * c ** 2 / Etotal_J * (Ez_tot + r * fidot * Br - rdot * Bfi)
                  - q * zdot / Etotal_J * (rdot * Er_tot + r * fidot * Efi_tot + zdot * Ez_tot))
    dxdt[:, 4] = fidot
    dxdt[:, 5] = q * (rdot * Er_tot + r * fidot * Efi_tot + zdot * Ez_tot)

    # 其余列快速清零
    dxdt[:, 6:] = 0

    return dxdt, Bz0, Br, Bfi


def func_get_R_B0(r, paras):
    q = paras.q
    f1 = paras.Bmap.f1
    Ek_value = paras.TempVariable
    P_start = FFAG_ConversionTools().Ek2P(Ek_value)
    y = r - P_start / q / f1(r)
    return y


if __name__ == '__main__':
    pass
