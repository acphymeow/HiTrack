import numpy as np
import re
import matplotlib.pyplot as plt
from mpi4py import MPI
import os
import copy
from scipy.interpolate import interp1d
from FFAG_MathTools import FFAG_interpolation, Lagrange_interp_2D_vect
from FFAG_ParasAndConversion import FFAG_ConversionTools, FFAG_GlobalParameters


class FFAG_Field:
    def __init__(self, filename):
        data = np.loadtxt(filename, skiprows=1)
        fi, r = data[0, 1:], data[1:, 0]
        self.r_axis, self.fi_axis, self.map_data = r, fi, data[1:, 1:]
        self.r_min, self.r_max, self.r_step = r[0], r[-1], r[1] - r[0]
        self.fi_min, self.fi_max, self.fi_step = fi[0], fi[-1], fi[1] - fi[0]
        self.r_size, self.fi_size = len(self.r_axis), len(self.fi_axis)


class FFAG_BField_new():

    def __init__(self, foldname, max_order, flag3D=True):
        self.max_order = max_order
        self.flag3D = flag3D

        # 初始化 Br, Bz, Bfi 的系数矩阵列表
        self.Br_coeff_matrices = []
        self.Bz_coeff_matrices = []
        self.Bfi_coeff_matrices = []

        # 检查 foldname 的类型并加载数据
        if isinstance(foldname, str):  # 如果 foldname 是路径字符串
            foldname = os.path.normpath(foldname)
            for order in range(max_order + 1):
                BrFileName = os.path.join(foldname, f"Br_order_{order}.npz")
                BzFileName = os.path.join(foldname, f"Bz_order_{order}.npz")
                BfiFileName = os.path.join(foldname, f"Bfi_order_{order}.npz")

                # 从 .npz 文件中加载各阶次系数矩阵
                self.Br_coeff_matrices.append(np.load(BrFileName)['Br_Taylor_coeff'])
                self.Bz_coeff_matrices.append(np.load(BzFileName)['Bz_Taylor_coeff'])
                self.Bfi_coeff_matrices.append(np.load(BfiFileName)['Bfi_Taylor_coeff'])

            # 设置 r 和 fi 的轴，从文件中读取数据
            self.BmapFileName = os.path.join(foldname, "Bmap.txt")
            data = np.loadtxt(self.BmapFileName, skiprows=1)
            fi, r = data[0, 1:], data[1:, 0]
            self.r_axis, self.fi_axis = r, fi
            self.map_data = self.Bz_coeff_matrices[0]

            # 读取 Bmap.txt 的表头以获取 Nsectors
            with open(self.BmapFileName, 'r') as file:
                match = re.search(r'Nsectors=(\d+)', file.readline().strip())
                self.Nsectors = int(match.group(1)) if match else None
        elif isinstance(foldname, np.ndarray):  # 如果 foldname 是 ndarray
            # 设置 r 和 fi 的轴，从数组中读取数据
            fi, r = foldname[0, 1:], foldname[1:, 0]
            self.r_axis, self.fi_axis = r, fi
            self.map_data = foldname[1:, 1:]
            self.Bz_coeff_matrices.append(self.map_data)
            self.Nsectors = None  # 如果是 ndarray 没有提供 Nsectors 信息
        else:
            raise TypeError("foldname 必须是路径字符串或 numpy.ndarray")

        # 通用属性计算
        self.r_min, self.r_max, self.r_step = r[0], r[-1], r[1] - r[0]
        self.fi_min, self.fi_max, self.fi_step = fi[0], fi[-1], fi[1] - fi[0]
        self.r_size, self.fi_size = len(self.r_axis), len(self.fi_axis)
        self.BMean = np.mean(self.map_data, axis=1)
        self.flutter = self.map_data / np.tile(self.BMean, (self.fi_size, 1)).T
        # self.f1 = FFAG_interpolation().linear_interpolation(self.r_axis, self.BMean)
        # 使用 interp1d 替换 FFAG_interpolation.linear_interpolation
        self.f1 = interp1d(self.r_axis, self.BMean, kind='linear', fill_value="extrapolate")
        self.rmin90 = self.r_min + (self.r_max - self.r_min) * 0.02
        Pmin90 = self.rmin90 * FFAG_GlobalParameters().q * self.f1(self.rmin90)
        self.rmax90 = self.r_max - (self.r_max - self.r_min) * 0.02
        Pmax90 = self.rmax90 * FFAG_GlobalParameters().q * self.f1(self.rmax90)
        self.Ekmin90 = FFAG_ConversionTools().P2Ek(Pmin90) / FFAG_GlobalParameters().q / 1e6
        self.Ekmax90 = FFAG_ConversionTools().P2Ek(Pmax90) / FFAG_GlobalParameters().q / 1e6

        # MPI 并行化打印信息
        comm = MPI.COMM_WORLD
        if comm.Get_rank() == 0:
            print('*' * 100)
            print(f'loading Bmap with order-{max_order + 1}- NonLinear terms ... ...')
            print('Bmap rmin = %.3fm, rmax = %.3fm' % (self.r_min, self.r_max))
            print(
                'Ek = %.3f MeV for R%.3fm orbit, Ek = %.3f MeV for R%.3fm orbit.' % (
                    self.Ekmin90, self.rmin90, self.Ekmax90, self.rmax90))
            print('*' * 100)

    def ExtendField(self, ext_Rsize_min, ext_Rsize_max, Binf):

        ext_min_arr = np.ones((ext_Rsize_min, self.fi_size)) * Binf * (-1)
        ext_max_arr = np.ones((ext_Rsize_max, self.fi_size)) * Binf

        r_min_axis_reverse = self.r_min - np.arange(1, ext_Rsize_min + 1, 1) * self.r_step
        r_max_axis = self.r_max + np.arange(1, ext_Rsize_max + 1, 1) * self.r_step
        r_min_axis = r_min_axis_reverse[::-1]

        r_axis_new = np.concatenate((r_min_axis, self.r_axis, r_max_axis))
        map_data_new = np.row_stack((ext_min_arr, self.map_data, ext_max_arr))
        data_without_fiAxis = np.column_stack((r_axis_new, map_data_new))

        data = np.row_stack((np.insert(self.fi_axis, 0, 0), data_without_fiAxis))

        fi, r = data[0, 1:], data[1:, 0]
        self.r_axis_ext, self.fi_axis_ext, self.map_data_ext = r, fi, data[1:, 1:]
        self.r_min_ext, self.r_max_ext, self.r_step_ext = r[0], r[-1], r[1] - r[0]
        self.fi_min_ext, self.fi_max_ext, self.fi_step_ext = fi[0], fi[-1], fi[1] - fi[0]
        self.r_size_ext, self.fi_size_ext = len(self.r_axis_ext), len(self.fi_axis_ext)

        self.BMean_ext = np.mean(self.map_data_ext, 1)
        # self.f1_ext = FFAG_interpolation().linear_interpolation(self.r_axis_ext, self.BMean_ext)
        # 替换 FFAG_interpolation.linear_interpolation
        self.f1_ext = interp1d(self.r_axis_ext, self.BMean_ext, kind='linear', fill_value="extrapolate")

    def Interpolation2DMap(self, r, fi, order, flag):
        """
        对某个阶次的矩阵进行插值
        flag: 0 -> Bz, 1 -> Br, 2 -> Bfi
        """
        if flag == 0:
            FieldMapTemp = self.Bz_coeff_matrices[order]
        elif flag == 1:
            FieldMapTemp = self.Bfi_coeff_matrices[order]
        elif flag == 2:
            FieldMapTemp = self.Br_coeff_matrices[order]
        elif flag == 3:
            FieldMapTemp = self.map_data_ext
        else:
            raise ValueError("flag 的值必须是 0（Bz）, 1（Br）, 2（Bfi）或 3（Bz_extend）")

        # 执行 2D 插值
        Value, OutRangeFlag = Lagrange_interp_2D_vect(self.r_axis, self.fi_axis, FieldMapTemp, r, fi)
        return Value, OutRangeFlag


class FFAG_EField:
    def __init__(self, EmapFoldName):
        self.Erfilename, self.Effilename, self.Ezfilename, self.CLfilename = (
            os.path.join(EmapFoldName, "Er.txt"), os.path.join(EmapFoldName, "Ef.txt"),
            os.path.join(EmapFoldName, "Ez.txt"), os.path.join(EmapFoldName, "CL.txt"))
        self.EmapFoldName = EmapFoldName
        data_Er = np.loadtxt(self.Erfilename, skiprows=1)
        data_Ef = np.loadtxt(self.Effilename, skiprows=1)
        data_Ez = np.loadtxt(self.Ezfilename, skiprows=1)
        data_Cl_RFi = np.loadtxt(self.CLfilename, skiprows=1)
        data_Cl_xy = np.zeros_like(data_Cl_RFi)
        data_Cl_xy[:, 0] = data_Cl_RFi[:, 0] * np.cos(data_Cl_RFi[:, 1])
        data_Cl_xy[:, 1] = data_Cl_RFi[:, 0] * np.sin(data_Cl_RFi[:, 1])

        fi, r = data_Er[0, 1:], data_Er[1:, 0]
        self.r_axis, self.fi_axis = r, fi
        self.r_min, self.r_max, self.r_step = r[0], r[-1], r[1] - r[0]
        self.fi_min, self.fi_max, self.fi_step = fi[0], fi[-1], fi[1] - fi[0]
        self.r_size, self.fi_size = len(self.r_axis), len(self.fi_axis)

        self.CentralLineRFi = data_Cl_RFi
        self.CentralLineXY = data_Cl_xy
        self.ErComponent = data_Er[1:, 1:]
        self.EfComponent = data_Ef[1:, 1:]
        self.EzComponent = data_Ez[1:, 1:]
        self.GetUnitinTxtFileEfield(self.Erfilename)  # change to default units
        self.Efield_period = None
        self.Efield_occupy = None
        self.FreqTimeCurve, self.FreqEkCurve, self.FixedFreqValue = None, None, None
        self.FreqEk_Ek, self.FreqEk_Freq = None, None
        self.SelfAdaptFreqValue = None
        self.VoltageTimeCurve, self.VoltageValue = None, None

    def GetUnitinTxtFileEfield(self, filename):
        with open(filename, 'r') as fid:
            unit_line = fid.readline()
            unitR_str = re.search(r'unitR=(\w+)', unit_line).group(1)
            unitF_str = re.search(r'unitFi=(\w+)', unit_line).group(1)
            unitE_str = re.search(r'unitE=(\w+)', unit_line).group(1)

            period_match = re.search(r"Period=(\d+)", unit_line)
            Efield_period = int(period_match.group(1))
            self.Efield_period = Efield_period

            occupy_match = re.search(r"occupy=([\d,]+)", unit_line)
            occupy_str = occupy_match.group(1)
            self.Efield_occupy = np.array([int(x) for x in occupy_str.split(',')])

            unitR = FFAG_GlobalParameters().units[unitR_str]
            unitF = FFAG_GlobalParameters().units[unitF_str]
            unitE = FFAG_GlobalParameters().units[unitE_str]
            self.r_axis = self.r_axis / unitR  # convert unit to meter
            self.fi_axis = self.fi_axis / unitF  # convert unit to rad
            self.ErComponent = self.ErComponent / unitE  # convert unit to V/m
            self.EfComponent = self.EfComponent / unitE  # convert unit to V/m
            self.EzComponent = self.EzComponent / unitE  # convert unit to V/m
            self.r_min, self.r_max, self.r_step = self.r_min / unitR, self.r_max / unitR, self.r_step / unitR
            self.fi_min, self.fi_max, self.fi_step = self.fi_min / unitF, self.fi_max / unitF, self.fi_step / unitF

    def Interpolation2DMapVectEfield(self, r, fi, flag):
        if flag == 0:
            FieldMapTemp = self.ErComponent
        elif flag == 1:
            FieldMapTemp = self.EfComponent
        else:
            FieldMapTemp = self.EzComponent
        # FFAG_interpolation_obj = FFAG_interpolation()
        # Value = FFAG_interpolation_obj.Lagrange_interp_2D_vect(
        #     self.r_axis, self.fi_axis, FieldMapTemp, r, fi)
        Value, _ = Lagrange_interp_2D_vect(
            self.r_axis, self.fi_axis, FieldMapTemp, r, fi)
        return Value

    def AddFixedFreqValue(self, FixedFreqValue):
        self.FixedFreqValue = FixedFreqValue

    def AddFreqEkCurve(self, SEOinfo):
        EkValuesAxis, FreqValuesAxis = SEOinfo[:, 1], SEOinfo[:, 4]
        # self.FreqEkCurve = FFAG_interpolation().linear_interpolation_vect(EkValuesAxis, FreqValuesAxis)
        self.FreqEk_Ek, self.FreqEk_Freq = EkValuesAxis, FreqValuesAxis

    def AddFreqTimeCurve(self, ):
        FreqTimeFilename = os.path.join(self.EmapFoldName, "RF_FreqCurve1002.rfc")
        FreqTimeCurve = np.loadtxt(FreqTimeFilename, skiprows=1)
        TimeAxis, RFFreqValue = FreqTimeCurve[:, 0], FreqTimeCurve[:, 1]
        self.FreqTimeCurve = FFAG_interpolation().linear_interpolation_vect(TimeAxis, RFFreqValue)

    def AddSelfAdaptFreq(self, ):
        pass

    def AddVoltageTimeCurve(self, VoltageTimeCurve):
        self.VoltageTimeCurve = VoltageTimeCurve

    def AddVoltageValue(self, VoltageValue):
        self.VoltageValue = VoltageValue

    def GetRFFreq_Ek(self, EkValueMeV):
        # w0 = 2*pi*f0
        return 2 * np.pi * self.FreqEkCurve(EkValueMeV)

    def GetRFFreq_time(self, RFtime):
        # w0 = 2*pi*f0
        return 2 * np.pi * self.FreqTimeCurve(np.array([RFtime, ]))

    def GetFixedFreqValue(self, ):
        # w0 = 2*pi*f0
        return 2 * np.pi * self.FixedFreqValue

    def GetVolt_Value(self, ):
        return self.VoltageValue
