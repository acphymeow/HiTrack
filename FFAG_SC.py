import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift
from numba import njit
import time


# from FFAG_MathTools import my_3dInterp_vect


@njit
def find_two_points_non_uniform_vect_njit(X, xi):
    right_point = np.searchsorted(X, xi, side='right')
    left_point = right_point - 1

    # Check if xi is out of the range of X
    idx_array_0, idx_array_1, flag = (
        left_point, right_point, np.zeros_like(xi))

    # if left_point < 0
    flag_left_point_less_0 = left_point < 0
    idx_array_0[flag_left_point_less_0] = 0
    idx_array_1[flag_left_point_less_0] = 1
    flag[flag_left_point_less_0] = 1

    # if right_right_point > len(X) - 1
    flag_right_point_larger_len = right_point > len(X) - 1
    idx_array_0[flag_right_point_larger_len] = len(X) - 2
    idx_array_1[flag_right_point_larger_len] = len(X) - 1
    flag[flag_right_point_larger_len] = 1

    idx_array = np.vstack((idx_array_0, idx_array_1))

    return idx_array, flag


@njit()
def my_3dInterp_vect(x, y, z, values, xi, yi, zi):
    x_idx, _ = find_two_points_non_uniform_vect_njit(x, xi)
    y_idx, _ = find_two_points_non_uniform_vect_njit(y, yi)
    z_idx, _ = find_two_points_non_uniform_vect_njit(z, zi)

    interp_values = xi * 0.0

    for idxs in range(len(xi)):
        xi_current, yi_current, zi_current = xi[idxs], yi[idxs], zi[idxs]

        x_left, x_right = x[x_idx[0, idxs]], x[x_idx[1, idxs]]
        y_left, y_right = y[y_idx[0, idxs]], y[y_idx[1, idxs]]
        z_left, z_right = z[z_idx[0, idxs]], z[z_idx[1, idxs]]

        x0_00 = values[x_idx[0, idxs], y_idx[0, idxs], z_idx[0, idxs]]
        x0_01 = values[x_idx[0, idxs], y_idx[0, idxs], z_idx[1, idxs]]
        x0_10 = values[x_idx[0, idxs], y_idx[1, idxs], z_idx[0, idxs]]
        x0_11 = values[x_idx[0, idxs], y_idx[1, idxs], z_idx[1, idxs]]

        x1_00 = values[x_idx[1, idxs], y_idx[0, idxs], z_idx[0, idxs]]
        x1_01 = values[x_idx[1, idxs], y_idx[0, idxs], z_idx[1, idxs]]
        x1_10 = values[x_idx[1, idxs], y_idx[1, idxs], z_idx[0, idxs]]
        x1_11 = values[x_idx[1, idxs], y_idx[1, idxs], z_idx[1, idxs]]

        x0_z0_interp = (yi_current - y_left) / (y_right - y_left) * x0_10 + (y_right - yi_current) / (
                    y_right - y_left) * x0_00
        x0_z1_interp = (yi_current - y_left) / (y_right - y_left) * x0_11 + (y_right - yi_current) / (
                    y_right - y_left) * x0_01
        x0_interp = (zi_current - z_left) / (z_right - z_left) * x0_z1_interp + (z_right - zi_current) / (
                    z_right - z_left) * x0_z0_interp

        x1_z0_interp = (yi_current - y_left) / (y_right - y_left) * x1_10 + (y_right - yi_current) / (
                    y_right - y_left) * x1_00
        x1_z1_interp = (yi_current - y_left) / (y_right - y_left) * x1_11 + (y_right - yi_current) / (
                    y_right - y_left) * x1_01
        x1_interp = (zi_current - z_left) / (z_right - z_left) * x1_z1_interp + (z_right - zi_current) / (
                    z_right - z_left) * x1_z0_interp

        interp_values[idxs] = (xi_current - x_left) / (x_right - x_left) * x1_interp + (x_right - xi_current) / (
                    x_right - x_left) * x0_interp

    return interp_values


@njit
def DistributeChargeNjit(idx_array_x, idx_array_y, idx_array_z, x_grid, y_grid, z_grid, charge_distribution,
                         gaussian_points):
    num_points = idx_array_x.shape[1]

    position_x = gaussian_points[:, 0]
    position_y = gaussian_points[:, 1]
    position_z = gaussian_points[:, 2]

    grid_spacing_x = x_grid[1] - x_grid[0]
    grid_spacing_y = y_grid[1] - y_grid[0]
    grid_spacing_z = z_grid[1] - z_grid[0]

    # 遍历每个散点，将每个散点的电荷量按距离加权分配到周围的8个顶点
    for index in range(num_points):
        x_index_0, x_index_1 = idx_array_x[0, index], idx_array_x[1, index]
        y_index_0, y_index_1 = idx_array_y[0, index], idx_array_y[1, index]
        z_index_0, z_index_1 = idx_array_z[0, index], idx_array_z[1, index]

        x0, x1 = x_grid[x_index_0], x_grid[x_index_1]
        y0, y1 = y_grid[y_index_0], y_grid[y_index_1]
        z0, z1 = z_grid[z_index_0], z_grid[z_index_1]

        wx0 = (x1 - position_x[index]) / grid_spacing_x
        wx1 = (position_x[index] - x0) / grid_spacing_x
        wy0 = (y1 - position_y[index]) / grid_spacing_y
        wy1 = (position_y[index] - y0) / grid_spacing_y
        wz0 = (z1 - position_z[index]) / grid_spacing_z
        wz1 = (position_z[index] - z0) / grid_spacing_z

        # 分配电荷到周围的8个顶点
        charge_distribution[x_index_0, y_index_0, z_index_0] += wx0 * wy0 * wz0
        charge_distribution[x_index_1, y_index_0, z_index_0] += wx1 * wy0 * wz0
        charge_distribution[x_index_0, y_index_1, z_index_0] += wx0 * wy1 * wz0
        charge_distribution[x_index_1, y_index_1, z_index_0] += wx1 * wy1 * wz0
        charge_distribution[x_index_0, y_index_0, z_index_1] += wx0 * wy0 * wz1
        charge_distribution[x_index_1, y_index_0, z_index_1] += wx1 * wy0 * wz1
        charge_distribution[x_index_0, y_index_1, z_index_1] += wx0 * wy1 * wz1
        charge_distribution[x_index_1, y_index_1, z_index_1] += wx1 * wy1 * wz1

    return charge_distribution


# Calculate Voltage for each grid
# in: a grid point in charge_distribution, out: voltage_distribution induced by this grid points
@njit
def CalculateVoltageForOneGrid(x_grid, y_grid, z_grid, charge_distribution, voltage_distribution, nml_index):
    epsilon_0 = 8.854187817e-12  # Permittivity of free space
    n_index, m_index, l_index = nml_index  # unpacking parameters

    # Source point coordinates and charge
    x0_source = x_grid[n_index]
    y0_source = y_grid[m_index]
    z0_source = z_grid[l_index]
    q_source = charge_distribution[n_index, m_index, l_index]

    # Grid dimensions
    n, m, l = charge_distribution.shape

    # Iterate over each grid point
    for i in range(n):
        for j in range(m):
            for k in range(l):
                # Avoid calculating the voltage at the source point itself
                if i == n_index and j == m_index and k == l_index:
                    continue

                # Calculate the distance from the source point to the current grid point
                r = np.sqrt((x_grid[i] - x0_source) ** 2 +
                            (y_grid[j] - y0_source) ** 2 +
                            (z_grid[k] - z0_source) ** 2) + 1e-10

                # Calculate the voltage contribution at the current grid point
                voltage_distribution[i, j, k] += (q_source / (4 * np.pi * epsilon_0 * r))

    return voltage_distribution


@njit()
def CalculateVoltageForAllGrids(x_grid, y_grid, z_grid, charge_distribution, voltage_distribution):
    # Grid dimensions
    n, m, l = charge_distribution.shape
    # Iterate over each grid point as a source point
    for n_index in range(n):
        for m_index in range(m):
            for l_index in range(l):
                if charge_distribution[n_index, m_index, l_index] != 0:
                    voltage_distribution_0 = voltage_distribution * 0.0
                    # Check if the grid point has charge
                    voltage_distribution_temp = CalculateVoltageForOneGrid(
                        x_grid, y_grid, z_grid, charge_distribution, voltage_distribution_0,
                        (n_index, m_index, l_index))
                    voltage_distribution = voltage_distribution + voltage_distribution_temp

    return voltage_distribution


# calculate static electric field for all grid points
@njit()
def CalculateEFieldFromVoltage(x_grid, y_grid, z_grid, voltage_distribution):
    n, m, l = voltage_distribution.shape
    Ez_distribution = np.zeros((n, m, l))
    Ex_distribution = np.zeros((n, m, l))
    Ey_distribution = np.zeros((n, m, l))

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    dz = z_grid[1] - z_grid[0]

    for i in range(1, n - 1):
        for j in range(1, m - 1):
            for k in range(1, l - 1):
                # Calculate the partial derivatives using central difference
                Ex_distribution[i, j, k] = -(voltage_distribution[i + 1, j, k] - voltage_distribution[i - 1, j, k]) / (
                        2 * dx)
                Ey_distribution[i, j, k] = -(voltage_distribution[i, j + 1, k] - voltage_distribution[i, j - 1, k]) / (
                        2 * dy)
                Ez_distribution[i, j, k] = -(voltage_distribution[i, j, k + 1] - voltage_distribution[i, j, k - 1]) / (
                        2 * dz)

        # 处理边界点
        # x方向边界
        for j in range(m):
            for k in range(l):
                Ex_distribution[0, j, k] = -(voltage_distribution[1, j, k] - voltage_distribution[0, j, k]) / dx
                Ex_distribution[n - 1, j, k] = -(
                        voltage_distribution[n - 1, j, k] - voltage_distribution[n - 2, j, k]) / dx

        # y方向边界
        for i in range(n):
            for k in range(l):
                Ey_distribution[i, 0, k] = -(voltage_distribution[i, 1, k] - voltage_distribution[i, 0, k]) / dy
                Ey_distribution[i, m - 1, k] = -(
                        voltage_distribution[i, m - 1, k] - voltage_distribution[i, m - 2, k]) / dy

        # z方向边界
        for i in range(n):
            for j in range(m):
                Ez_distribution[i, j, 0] = -(voltage_distribution[i, j, 1] - voltage_distribution[i, j, 0]) / dz
                Ez_distribution[i, j, l - 1] = -(
                        voltage_distribution[i, j, l - 1] - voltage_distribution[i, j, l - 2]) / dz

    return Ex_distribution, Ey_distribution, Ez_distribution


# @njit
def SC_calculator(gaussian_points, percentile_low, percentile_high, n, m, l):
    # # Using percentiles to remove outliers and determine the range
    # percentile_low = 0
    # percentile_high = 100
    # 定义常数
    epsilon_0 = 8.854187817e-12  # 真空介电常数

    x_min, x_max = np.percentile(gaussian_points[:, 0], [percentile_low, percentile_high])
    y_min, y_max = np.percentile(gaussian_points[:, 1], [percentile_low, percentile_high])
    z_min, z_max = np.percentile(gaussian_points[:, 2], [percentile_low, percentile_high])

    # # Defining the grid size
    # n, m, l = 40, 40, 40  # Number of divisions in x, y, and z directions

    # Generating grid points
    x_grid = np.linspace(x_min, x_max, n)
    y_grid = np.linspace(y_min, y_max, m)
    z_grid = np.linspace(z_min, z_max, l)

    # Initializing a matrix to store the charge distribution
    charge_distribution = np.zeros((n, m, l))
    voltage_distribution = np.zeros((n, m, l))

    point_x, point_y, point_z = gaussian_points[:, 0], gaussian_points[:, 1], gaussian_points[:, 2]
    idx_array_x, _ = find_two_points_non_uniform_vect_njit(x_grid, point_x)
    idx_array_y, _ = find_two_points_non_uniform_vect_njit(y_grid, point_y)
    idx_array_z, _ = find_two_points_non_uniform_vect_njit(z_grid, point_z)

    start_time = time.time()
    charge_distribution_out = DistributeChargeNjit(idx_array_x, idx_array_y, idx_array_z,
                                                   x_grid, y_grid, z_grid,
                                                   charge_distribution, gaussian_points)
    # charge_distribution_out[n // 2, m // 2, l // 2] = 1.0
    end_time = time.time()
    print(f"track time 1 = {end_time - start_time}")

    start_time = time.time()
    voltage_distribution_out = CalculateVoltageForAllGrids(x_grid, y_grid, z_grid,
                                                           charge_distribution_out, voltage_distribution)
    end_time = time.time()
    print(f"track time 2 = {end_time - start_time}")

    start_time = time.time()
    Ex_distribution, Ey_distribution, Ez_distribution = (
        CalculateEFieldFromVoltage(x_grid, y_grid, z_grid, voltage_distribution_out))
    end_time = time.time()
    print(f"track time 3 = {end_time - start_time}")

    start_time = time.time()
    Ex_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ex_distribution, point_x, point_y, point_z)
    Ey_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ey_distribution, point_x, point_y, point_z)
    Ez_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ez_distribution, point_x, point_y, point_z)
    end_time = time.time()
    print(f"track time 4 = {end_time - start_time}")

    return (Ex_distribution, Ey_distribution, Ez_distribution, Ex_interp, Ey_interp, Ez_interp,
            voltage_distribution_out, charge_distribution_out, x_grid, y_grid, z_grid)


def green_function(x, y, z):
    epsilon = 1e-8  # 避免除以零
    return 1.0 / (np.sqrt(x ** 2 + y ** 2 + z ** 2) + epsilon)


def SC_calculator_fft(gaussian_points, percentile_low, percentile_high, n, m, l):
    # # Using percentiles to remove outliers and determine the range
    # percentile_low = 0
    # percentile_high = 100
    # 定义常数
    epsilon_0 = 8.854187817e-12  # 真空介电常数

    x_min, x_max = np.percentile(gaussian_points[:, 0], [percentile_low, percentile_high])
    y_min, y_max = np.percentile(gaussian_points[:, 1], [percentile_low, percentile_high])
    z_min, z_max = np.percentile(gaussian_points[:, 2], [percentile_low, percentile_high])

    # # Defining the grid size
    # Generating grid points
    x_grid = np.linspace(x_min, x_max, n)
    y_grid = np.linspace(y_min, y_max, m)
    z_grid = np.linspace(z_min, z_max, l)
    Xmatrix, Ymatrix, Zmatrix = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    G_function = green_function(Xmatrix, Ymatrix, Zmatrix)
    # G_fft = fftn(fftshift(G_function))
    G_fft = fftn(G_function)

    # Initializing a matrix to store the charge distribution
    charge_distribution = np.zeros((n, m, l))
    # voltage_distribution_out = np.zeros((n, m, l))

    point_x, point_y, point_z = gaussian_points[:, 0], gaussian_points[:, 1], gaussian_points[:, 2]
    idx_array_x, _ = find_two_points_non_uniform_vect_njit(x_grid, point_x)
    idx_array_y, _ = find_two_points_non_uniform_vect_njit(y_grid, point_y)
    idx_array_z, _ = find_two_points_non_uniform_vect_njit(z_grid, point_z)

    start_time = time.time()
    charge_distribution_out = DistributeChargeNjit(idx_array_x, idx_array_y, idx_array_z,
                                                   x_grid, y_grid, z_grid,
                                                   charge_distribution, gaussian_points)
    # charge_distribution_out[n // 2, m // 2, l // 2] = 1.0/8.0
    # charge_distribution_out[n // 2, m // 2+1, l // 2] = 1.0/8.0
    # charge_distribution_out[n // 2+1, m // 2, l // 2] = 1.0/8.0
    # charge_distribution_out[n // 2+1, m // 2+1, l // 2] = 1.0/8.0
    # charge_distribution_out[n // 2, m // 2, l // 2+1] = 1.0/8.0
    # charge_distribution_out[n // 2, m // 2+1, l // 2+1] = 1.0/8.0
    # charge_distribution_out[n // 2+1, m // 2, l // 2+1] = 1.0/8.0
    # charge_distribution_out[n // 2+1, m // 2+1, l // 2+1] = 1.0/8.0

    rho_fft = fftn(charge_distribution_out)
    # 在频域中相乘
    voltage_distribution_fft = G_fft * rho_fft * 1 / (4 * np.pi * epsilon_0)
    # voltage_distribution_out = np.real(ifftn(voltage_distribution_fft))
    # 结果中的频率或时间分量的顺序不同, 默认情况下是将零频率或零时间分量放在数组的起始位置。
    voltage_distribution_out = np.real(fftshift(ifftn(voltage_distribution_fft)))
    end_time = time.time()
    print(f"track time 1 = {end_time - start_time}")

    start_time = time.time()
    Ex_distribution, Ey_distribution, Ez_distribution = (
        CalculateEFieldFromVoltage(x_grid, y_grid, z_grid, voltage_distribution_out))
    end_time = time.time()
    print(f"track time 3 = {end_time - start_time}")

    start_time = time.time()
    Ex_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ex_distribution, point_x, point_y, point_z)
    Ey_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ey_distribution, point_x, point_y, point_z)
    Ez_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ez_distribution, point_x, point_y, point_z)
    end_time = time.time()
    print(f"track time 4 = {end_time - start_time}")

    return (Ex_distribution, Ey_distribution, Ez_distribution, Ex_interp, Ey_interp, Ez_interp,
            voltage_distribution_out, charge_distribution_out, x_grid, y_grid, z_grid)




def ConvertToPolarElectricField(point_x, point_y, Ex_interp, Ey_interp, Ez_interp):
    """
    将插值后的电场分量从笛卡尔坐标 (Ex, Ey, Ez) 转换为极坐标系下的电场分量 (Er, Efi, Ez)。

    参数：
    - point_x, point_y: 粒子的位置坐标 (x, y)
    - Ex_interp, Ey_interp, Ez_interp: 插值得到的笛卡尔坐标电场分量 (Ex, Ey, Ez)

    返回：
    - Er_interp, Efi_interp, Ez_interp: 极坐标下的电场分量 (Er, Efi, Ez)
    """
    # 计算粒子的径向距离 r 和方位角 phi
    phi = np.arctan2(point_y, point_x)

    # 计算极坐标下的径向电场 Er 和方位角电场 Efi
    Er_interp = Ex_interp * np.cos(phi) + Ey_interp * np.sin(phi)
    Efi_interp = -Ex_interp * np.sin(phi) + Ey_interp * np.cos(phi)

    # Ez_interp 保持不变
    return Er_interp, Ez_interp, Efi_interp


def SC_calculator_fft_new(charge_distribution_out,
                          x_grid, y_grid, z_grid,
                          Xmatrix, Ymatrix, Zmatrix,
                          point_x, point_y, point_z):
    """
    使用FFT计算空间电荷场，输入为已知的3维电荷分布矩阵和3维网格矩阵。

    参数：
    - charge_distribution_out: numpy.ndarray
        包含网格点上电荷分布的3D数组，表示空间中的电荷密度分布。
    - x_grid, y_grid, z_grid: numpy.ndarray
        网格在x, y, z方向的坐标数组。
    - Xmatrix, Ymatrix, Zmatrix: numpy.ndarray
        通过x_grid, y_grid, z_grid生成的3D网格坐标矩阵，表示网格中的点的三维坐标。
    - point_x, point_y, point_z: numpy.ndarray
        粒子在x, y, z方向上的坐标数组，用于插值电场值到粒子位置。

    返回：
    - Ex_distribution, Ey_distribution, Ez_distribution: numpy.ndarray
        在x, y, z方向的电场分布，表示在网格上每个点的电场值。
    - Ex_interp, Ey_interp, Ez_interp: numpy.ndarray
        在粒子位置插值得到的x, y, z方向的电场值。
    - voltage_distribution_out: numpy.ndarray
        在网格上计算得到的电势分布。
    - charge_distribution_out: numpy.ndarray
        电荷分布矩阵，作为函数的输入返回。
    - x_grid, y_grid, z_grid: numpy.ndarray
        网格点的坐标。

    流程：
    1. 计算格林函数：使用空间坐标矩阵 (Xmatrix, Ymatrix, Zmatrix) 计算格林函数 G。
    2. 使用FFT计算电势分布：
       - 对电荷分布 `charge_distribution_out` 进行傅里叶变换得到频域电荷分布 `rho_fft`。
       - 通过格林函数的傅里叶变换 `G_fft` 和电荷分布的傅里叶变换计算电势的傅里叶表示。
       - 反傅里叶变换得到空间电势分布 `voltage_distribution_out`。
    3. 计算电场分布：
       - 通过电势分布计算网格上各点的电场 `Ex_distribution`, `Ey_distribution`, `Ez_distribution`。
    4. 插值电场到粒子位置：
       - 使用 `my_3dInterp_vect` 对网格电场进行插值，得到粒子位置的电场值 `Ex_interp`, `Ey_interp`, `Ez_interp`。
    5. 返回电场分布、电势分布及网格点信息。
    """

    epsilon_0 = 8.854187817e-12  # 真空介电常数

    # 计算格林函数
    G_function = green_function(Xmatrix-np.mean(x_grid), Ymatrix-np.mean(y_grid), Zmatrix-np.mean(z_grid))
    G_fft = fftn(G_function)

    # 计算电势分布
    rho_fft = fftn(charge_distribution_out)
    # 在频域中相乘
    voltage_distribution_fft = G_fft * rho_fft * 1 / (4 * np.pi * epsilon_0)
    # 结果中的频率或时间分量的顺序不同, 默认情况下是将零频率或零时间分量放在数组的起始位置。
    voltage_distribution_out = fftshift(np.real(ifftn(voltage_distribution_fft)))

    # rho_fft = fftn(charge_distribution_out)
    # # 在频域中相乘
    # voltage_distribution_fft = G_fft * rho_fft * 1 / (4 * np.pi * epsilon_0)
    # # voltage_distribution_out = np.real(ifftn(voltage_distribution_fft))
    # # 结果中的频率或时间分量的顺序不同, 默认情况下是将零频率或零时间分量放在数组的起始位置。
    # voltage_distribution_out = np.real(fftshift(ifftn(voltage_distribution_fft)))

    # 计算电场分布
    Ex_distribution, Ey_distribution, Ez_distribution = (
        CalculateEFieldFromVoltage(x_grid, y_grid, z_grid, voltage_distribution_out))

    # 插值电场到粒子位置
    Ex_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ex_distribution, point_x, point_y, point_z)
    Ey_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ey_distribution, point_x, point_y, point_z)
    Ez_interp = my_3dInterp_vect(x_grid, y_grid, z_grid, Ez_distribution, point_x, point_y, point_z)

    # 将插值得到的电场分量从笛卡尔坐标 (Ex_interp, Ey_interp, Ez_interp) 转换为极坐标系 (Er, Efi, Ez)
    Er_interp, Efi_interp, Ez_interp = ConvertToPolarElectricField(point_x, point_y, Ex_interp, Ey_interp, Ez_interp)

    return (Ex_distribution, Ey_distribution, Ez_distribution, Er_interp, Ez_interp, Efi_interp,
            voltage_distribution_out, charge_distribution_out, x_grid, y_grid, z_grid)


def Bunch_SC_Calculator(Bunch_obj):
    """
    使用FFT方法计算Bunch_obj中所有粒子的自感应空间电荷电场。

    参数：
    - Bunch_obj: FFAG_Bunch 对象
        表示粒子束的对象，包含粒子的位置信息、电荷分布以及网格信息。

    流程：
    1. 检查网格信息是否存在。如果网格信息为空，则不进行计算。
    2. 提取存活粒子的坐标。
    3. 调用 SC_calculator_fft_new 函数计算电场分布，并将电场值插值到粒子位置。
    4. 更新 Bunch_obj 中的局部电场分量。

    返回：
    - Ex_interp, Ey_interp, Ez_interp: numpy.ndarray
        存活粒子位置的电场插值值。
    """

    # 判断 Bunch_obj 中的网格信息是否存在，如果为空则直接返回
    if Bunch_obj.xmax_Global is None or Bunch_obj.zmax_Global is None or Bunch_obj.Zgrid is None:
        print("Grid information is missing. Cannot calculate space charge electric field.")
        return None, None, None, None, None, None, None, None

    # 提取 FFAG_Bunch 中存活粒子的笛卡尔坐标
    local_points, Local_ID, Global_ID = Bunch_obj.GetLocalCoordinates()

    # 从 Bunch_obj 中提取电荷分布和网格信息
    charge_distribution_out = Bunch_obj.charge_distribution_global  # 全局电荷分布矩阵
    x_grid, y_grid, z_grid = Bunch_obj.Xgrid, Bunch_obj.Ygrid, Bunch_obj.Zgrid  # 网格坐标
    Xmatrix, Ymatrix, Zmatrix = Bunch_obj.Xmatrix, Bunch_obj.Ymatrix, Bunch_obj.Zmatrix  # 3D 网格矩阵

    # 提取存活粒子的 x, y, z 坐标
    point_x, point_y, point_z = local_points[:, 0], local_points[:, 1], local_points[:, 2]

    # 使用 SC_calculator_fft_new 计算空间电荷场并进行电场插值
    (Ex_distribution, Ey_distribution, Ez_distribution,
     Er_interp, Ez_interp, Efi_interp,
     voltage_distribution_out, charge_distribution_out, _, _, _) = (
        SC_calculator_fft_new(charge_distribution_out,
                              x_grid, y_grid, z_grid,
                              Xmatrix, Ymatrix, Zmatrix,
                              point_x, point_y, point_z)
    )

    # 将插值后的电场值存储到 Bunch_obj 中，更新 LocalBunch 的电场分量
    Bunch_obj.Update_SC_Efield_Local(Er_interp, Ez_interp, Efi_interp, Local_ID)

    return (Ex_distribution, Ey_distribution, Ez_distribution,
            Er_interp, Efi_interp, Ez_interp,
            voltage_distribution_out, charge_distribution_out)


if __name__ == '__main__':

    # Parameters for Gaussian distribution
    num_points = 100000
    mean = [0, 0, 0]  # mean at the origin
    cov = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # diagonal covariance, points are not correlated
    # # 设置随机数种子
    # np.random.seed(42)
    # Generating Gaussian distributed points
    gaussian_points = np.random.multivariate_normal(mean, cov, num_points)
    gaussian_points[:, 0] += 5

    # Using percentiles to remove outliers and determine the range
    percentile_low = 0
    percentile_high = 100

    # Defining the grid size
    n, m, l = 40, 40, 40  # Number of divisions in x, y, and z directions
    SC_calculator(gaussian_points, percentile_low, percentile_high, n, m, l)

    (Ex_distribution_direct, Ey_distribution_direct, Ez_distribution_direct,
     Ex_interp, Ey_interp, Ez_interp,
     voltage_distribution_direct, charge_distribution_direct, x_grid_direct, y_grid_direct, z_grid_direct) = (
        SC_calculator(gaussian_points, percentile_low, percentile_high, n, m, l))

    # Defining the grid size
    SC_calculator_fft(gaussian_points, percentile_low, percentile_high, n, m, l)

    (Ex_distribution_fft, Ey_distribution_fft, Ez_distribution_fft,
     Ex_interp_fft, Ey_interp_fft, Ez_interp_fft,
     voltage_distribution_fft, charge_distribution_fft, x_grid_fft, y_grid_fft, z_grid_fft) = (
        SC_calculator_fft(gaussian_points, percentile_low, percentile_high, n, m, l))


    xx_mesh_direct, yy_mesh_direct = np.meshgrid(x_grid_direct, y_grid_direct)
    xx_mesh_fft, yy_mesh_fft = np.meshgrid(x_grid_fft, y_grid_fft)

    # visualization
    # 选择中间层进行剖面图的绘制
    xy_slice_direct = voltage_distribution_direct[:, :, l // 3 * 1]  # XY平面
    xz_slice_direct = voltage_distribution_direct[:, :, l // 2 * 1]  # XZ平面
    yz_slice_direct = voltage_distribution_direct[:, :, l // 3 * 2]  # YZ平面
    xy_slice_fft = voltage_distribution_fft[:, :, l // 3 * 1]  # XY平面
    xz_slice_fft = voltage_distribution_fft[:, :, l // 2 * 1]  # XZ平面
    yz_slice_fft = voltage_distribution_fft[:, :, l // 3 * 2]  # YZ平面

    # Modifying the visualization to use the same color bar for all three plots for comparison

    # Find the global minimum and maximum voltage values for consistent color mapping
    vmin = np.min(voltage_distribution_direct)
    vmax = np.max(voltage_distribution_direct)
    vmin_fft = np.min(voltage_distribution_fft)
    vmax_fft = np.max(voltage_distribution_fft)

    fig, axs = plt.subplots(2, 3, figsize=(9, 5))

    # 绘制_direct的结果
    for i, slice_data in enumerate([xy_slice_direct, xz_slice_direct, yz_slice_direct]):
        im = axs[0, i].imshow(slice_data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        # axs[0, i].set_title(f'_direct Voltage Distribution at {["z/3", "z/2", "2z/3"][i]}')
        axs[0, i].set_xlabel('X axis')
        axs[0, i].set_ylabel('Y axis')

    # 绘制_fft的结果
    for i, slice_data in enumerate([xy_slice_fft, xz_slice_fft, yz_slice_fft]):
        im = axs[1, i].imshow(slice_data, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
        # axs[1, i].set_title(f'_fft Voltage Distribution at {["z/3", "z/2", "2z/3"][i]}')
        axs[1, i].set_xlabel('X axis')
        axs[1, i].set_ylabel('Y axis')

    # 为所有图像创建一个统一的颜色条
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.2, label='Voltage (V)')
    plt.subplots_adjust(right=0.85)
    # plt.tight_layout()

    # fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    # # Plotting each slice with the same color scale
    # cmap = 'viridis'  # Color map
    # for i, slice_data in enumerate([xy_slice_direct, xz_slice_direct, yz_slice_direct]):
    #     pos = axs[i].imshow(slice_data, cmap=cmap, origin='lower')
    #     axs[i].set_title(f'Voltage Distribution at {["z/5", "z/2", "4z/5"][i]}')
    #     axs[i].set_xlabel('X axis')
    #     axs[i].set_ylabel('Y axis')
    # # Create a single color bar for all plots
    # fig.colorbar(pos, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, label='Voltage (V)')

    # vx_slice_direct = voltage_distribution_direct[:, m // 2, l // 3 * 1]  # XY平面
    # vx_slice_fft = voltage_distribution_fft[:, m // 2, l // 3 * 1]  # XY平面
    # plt.figure()
    # plt.plot(vx_slice_direct)
    # plt.plot(vx_slice_fft)

    ra1 = np.arange(-5.0, -1.0, 0.01)
    ra2 = np.arange(1.0, 5.0, 0.01)
    epsilon_0 = 8.854187817e-12  # 真空介电常数
    E_theory1 = (1 / (4 * np.pi * epsilon_0)) / (ra1) ** 2 * (ra1 / np.abs(ra1))
    E_theory2 = (1 / (4 * np.pi * epsilon_0)) / (ra2) ** 2 * (ra2 / np.abs(ra2))

    ex_slice_direct = Ex_distribution_direct[:, m // 2, l // 2]  # XY平面
    ex_slice_fft = Ex_distribution_fft[:, m // 2, l // 2]  # XY平面
    plt.figure()
    plt.plot(x_grid_direct, ex_slice_direct, label="PIC")
    plt.plot(x_grid_direct, ex_slice_fft, linewidth=2.5, label="PIC-FFT")
    # plt.plot(ra2, E_theory2, linestyle='dashed', color='red', linewidth=2.5, label="theoretical")
    # plt.plot(ra1, E_theory1, linestyle='dashed', color='red', linewidth=2.5)
    plt.xlabel('X(m)')
    plt.ylabel('Ex(V/m)')
    plt.legend(loc='best')
    fig, axs = plt.subplots(2, 3, figsize=(9, 5))
    # 绘制_direct的结果
    for i, slice_data in enumerate([xy_slice_direct, xz_slice_direct, yz_slice_direct]):
        axs[0, i].streamplot(xx_mesh_fft, yy_mesh_fft, Ey_distribution_fft[:, :, l // 2],
                             Ex_distribution_fft[:, :, l // 2], color='b', linewidth=1, density=2)
        # axs[0, i].set_xlabel('X axis')
        if i == 0:
            axs[0, i].set_ylabel('Y axis')

    # 绘制_fft的结果
    for i, slice_data in enumerate([xy_slice_fft, xz_slice_fft, yz_slice_fft]):
        axs[1, i].streamplot(xx_mesh_direct, yy_mesh_direct, Ey_distribution_direct[:, :, l // 2],
                             Ex_distribution_direct[:, :, l // 2], color='b', linewidth=1, density=2)
        axs[1, i].set_xlabel('X axis')
        if i == 0:
            axs[1, i].set_ylabel('Y axis')

    plt.show()
