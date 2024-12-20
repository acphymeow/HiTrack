import numpy as np
import matplotlib.pyplot as plt
import time
import math
import copy


def process_ndarray(x0_raw):
    if x0_raw.ndim == 1:
        x0 = x0_raw
    elif x0_raw.ndim == 2:
        x0 = x0_raw.flatten()
    else:
        raise ValueError("Input ndarray must be 1D or 2D")
    return x0


def restore_shape(x0_processed, original_shape):
    if len(original_shape) == 1:
        return x0_processed
    elif len(original_shape) == 2:
        return x0_processed.reshape(original_shape)
    else:
        raise ValueError("Original shape must be 1D or 2D.")


def find_two_points_non_uniform_vector(X, xi):
    """
    A vectorized function to find the nearest two sample points around xi
    and return their positions.
    """
    right_point = np.searchsorted(X, xi, side='right')
    left_point = right_point - 1
    # Check if xi is out of the range of X
    flag_left_point_less_than_0 = left_point < 0
    flag_right_point_larger_than_len = right_point > len(X) - 1
    idx_array_0, idx_array_1, flag = copy.deepcopy(left_point), copy.deepcopy(right_point), np.zeros_like(xi)
    idx_array_0[flag_left_point_less_than_0] = 0
    idx_array_1[flag_left_point_less_than_0] = 1
    flag[flag_left_point_less_than_0] = 1

    idx_array_0[flag_right_point_larger_than_len] = len(X) - 2
    idx_array_1[flag_right_point_larger_than_len] = len(X) - 1
    flag[flag_right_point_larger_than_len] = 1

    idx_array = np.dstack((idx_array_0, idx_array_1))

    return idx_array, flag


def find_four_points_non_uniform_vector(X, xi):
    """
    Find the nearest 4 sample points around xi and return their positions.
    This is a vectorized function.
    """
    right_point = np.searchsorted(X, xi, side='right')
    left_point = right_point - 1
    right_right_point = right_point + 1
    left_left_point = left_point - 1
    # Check if xi is out of the range of X
    flag_left_point_less_than_0 = left_left_point < 0
    flag_right_point_larger_than_len = right_right_point > len(X) - 1
    idx_array_0, idx_array_1, idx_array_2, idx_array_3, flag = \
        copy.deepcopy(left_left_point), copy.deepcopy(left_point), \
        copy.deepcopy(right_point), copy.deepcopy(right_right_point), \
        np.zeros_like(xi)
    # if left_left_point < 0
    idx_array_0[flag_left_point_less_than_0] = 0
    idx_array_1[flag_left_point_less_than_0] = 1
    idx_array_2[flag_left_point_less_than_0] = 2
    idx_array_3[flag_left_point_less_than_0] = 3
    flag[flag_left_point_less_than_0] = 1

    # if right_right_point > len(X) - 1
    idx_array_0[flag_right_point_larger_than_len] = len(X) - 4
    idx_array_1[flag_right_point_larger_than_len] = len(X) - 3
    idx_array_2[flag_right_point_larger_than_len] = len(X) - 2
    idx_array_3[flag_right_point_larger_than_len] = len(X) - 1
    flag[flag_right_point_larger_than_len] = 1

    idx_array = np.dstack((idx_array_0, idx_array_1, idx_array_2, idx_array_3))

    return idx_array, flag


def calculate_4x4_determinant(matrix):
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]  # a~d:i*j matrix
    e, f, g, h = matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]
    i, j, k, l = matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]
    m, n, o, p = matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]

    determinant = a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n)) \
                  - b * (e * (k * p - l * o) - g * (i * p - l * m) + h * (i * o - k * m)) \
                  + c * (e * (j * p - l * n) - f * (i * p - l * m) + h * (i * n - j * m)) \
                  - d * (e * (j * o - k * n) - f * (i * o - k * m) + g * (i * n - j * m))
    return determinant


def linear_interpolation_vectorization(X, Y):
    # dimension of XY: 1*n and 1*n, fi and r of the spiral
    # dimension of xi: i*j, fi of the nearest point on the spiral for each grid points.
    # x0~x3:i*j matrix, 4 matrix of fi of the 4 nearest points around xi
    # y0~y3:i*j matrix, 4 matrix of r of the 4 nearest points around xi
    def interpolator_vector(xi):
        idx_array, _ = find_two_points_non_uniform_vector(X, xi)
        idx_array_0, idx_array_1 = idx_array[:, :, 0], idx_array[:, :, 1]
        x0, x1, y0, y1 = X[idx_array_0], X[idx_array_1], Y[idx_array_0], Y[idx_array_1]
        return y0 + (xi - x0) / (x1 - x0) * (y1 - y0)

    def interpolator_vector_spline(xi):
        idx_array_derv, _ = find_four_points_non_uniform_vector(X, xi)
        idx_array_derv_0, idx_array_derv_1, idx_array_derv_2, idx_array_derv_3 = \
            idx_array_derv[:, :, 0], idx_array_derv[:, :, 1], idx_array_derv[:, :, 2], idx_array_derv[:, :, 3]
        x0, x1, x2, x3 = X[idx_array_derv_0], X[idx_array_derv_1], X[idx_array_derv_2], X[idx_array_derv_3]
        y0, y1, y2, y3 = Y[idx_array_derv_0], Y[idx_array_derv_1], Y[idx_array_derv_2], Y[idx_array_derv_3]
        # [[x0**3,x0**2,x0,1], [[a],  [[y0],
        #  [x1**3,x1**2,x1,1],  [b],   [y1]
        #  [x2**3,x2**2,x2,1],  [c],   [y2]
        #  [x3**3,x3**2,x3,1]]* [d]] = [y3]]
        MatrixA = [[x0 ** 3, x0 ** 2, x0, 1], [x1 ** 3, x1 ** 2, x1, 1], [x2 ** 3, x2 ** 2, x2, 1],
                   [x3 ** 3, x3 ** 2, x3, 1]]
        Matrix_a = [[y0, x0 ** 2, x0, 1], [y1, x1 ** 2, x1, 1], [y2, x2 ** 2, x2, 1], [y3, x3 ** 2, x3, 1]]
        Matrix_b = [[x0 ** 3, y0, x0, 1], [x1 ** 3, y1, x1, 1], [x2 ** 3, y2, x2, 1], [x3 ** 3, y3, x3, 1]]
        Matrix_c = [[x0 ** 3, x0 ** 2, y0, 1], [x1 ** 3, x1 ** 2, y1, 1], [x2 ** 3, x2 ** 2, y2, 1],
                    [x3 ** 3, x3 ** 2, y3, 1]]
        Matrix_d = [[x0 ** 3, x0 ** 2, x0, y0], [x1 ** 3, x1 ** 2, x1, y1], [x2 ** 3, x2 ** 2, x2, y2],
                    [x3 ** 3, x3 ** 2, x3, y3]]
        DetA = calculate_4x4_determinant(MatrixA)
        Det_a = calculate_4x4_determinant(Matrix_a)
        Det_b = calculate_4x4_determinant(Matrix_b)
        Det_c = calculate_4x4_determinant(Matrix_c)
        Det_d = calculate_4x4_determinant(Matrix_d)

        a, b, c, d = Det_a / DetA, Det_b / DetA, Det_c / DetA, Det_d / DetA
        return a * xi ** 3 + b * xi**2 + c*xi+d

    def derv_interpolator_vector(xi):
        idx_array_derv, _ = find_four_points_non_uniform_vector(X, xi)
        idx_array_derv_0, idx_array_derv_1, idx_array_derv_2, idx_array_derv_3 = \
            idx_array_derv[:, :, 0], idx_array_derv[:, :, 1], idx_array_derv[:, :, 2], idx_array_derv[:, :, 3]
        x0, x1, x2, x3 = X[idx_array_derv_0], X[idx_array_derv_1], X[idx_array_derv_2], X[idx_array_derv_3]
        y0, y1, y2, y3 = Y[idx_array_derv_0], Y[idx_array_derv_1], Y[idx_array_derv_2], Y[idx_array_derv_3]
        # [[x0**3,x0**2,x0,1], [[a],  [[y0],
        #  [x1**3,x1**2,x1,1],  [b],   [y1]
        #  [x2**3,x2**2,x2,1],  [c],   [y2]
        #  [x3**3,x3**2,x3,1]]* [d]] = [y3]]
        MatrixA = [[x0 ** 3, x0 ** 2, x0, 1], [x1 ** 3, x1 ** 2, x1, 1], [x2 ** 3, x2 ** 2, x2, 1],
                   [x3 ** 3, x3 ** 2, x3, 1]]
        Matrix_a = [[y0, x0 ** 2, x0, 1], [y1, x1 ** 2, x1, 1], [y2, x2 ** 2, x2, 1], [y3, x3 ** 2, x3, 1]]
        Matrix_b = [[x0 ** 3, y0, x0, 1], [x1 ** 3, y1, x1, 1], [x2 ** 3, y2, x2, 1], [x3 ** 3, y3, x3, 1]]
        Matrix_c = [[x0 ** 3, x0 ** 2, y0, 1], [x1 ** 3, x1 ** 2, y1, 1], [x2 ** 3, x2 ** 2, y2, 1],
                    [x3 ** 3, x3 ** 2, y3, 1]]
        Matrix_d = [[x0 ** 3, x0 ** 2, x0, y0], [x1 ** 3, x1 ** 2, x1, y1], [x2 ** 3, x2 ** 2, x2, y2],
                    [x3 ** 3, x3 ** 2, x3, y3]]
        DetA = calculate_4x4_determinant(MatrixA)
        Det_a = calculate_4x4_determinant(Matrix_a)
        Det_b = calculate_4x4_determinant(Matrix_b)
        Det_c = calculate_4x4_determinant(Matrix_c)
        Det_d = calculate_4x4_determinant(Matrix_d)

        a, b, c, d = Det_a / DetA, Det_b / DetA, Det_c / DetA, Det_d / DetA
        return 3 * a * xi ** 2 + 2 * b * xi + c

    return interpolator_vector, derv_interpolator_vector, interpolator_vector_spline


def find_quadratic_min(x0_raw, y0_raw, x1_raw, y1_raw, x2_raw, y2_raw):
    """
    :param: xy coordinates of the nearest three points on the spiral iterated on each grid point. x0_raw is the
    x(or theta) coordinate of the first point, y coordinate is the distance.
    :return: i*j ndarray
    """
    x0, y0 = process_ndarray(x0_raw), process_ndarray(y0_raw)
    x1, y1 = process_ndarray(x1_raw), process_ndarray(y1_raw)
    x2, y2 = process_ndarray(x2_raw), process_ndarray(y2_raw)

    Delta_a = y0 * x1 * 1 + x0 * 1 * y2 + 1 * y1 * x2 - y0 * 1 * x2 - x0 * y1 * 1 - 1 * x1 * y2
    DetA = x0 ** 2 * x1 * 1 + x0 * 1 * x2 ** 2 + 1 * x1 ** 2 * x2 - x0 ** 2 * 1 * x2 - x0 * x1 ** 2 * 1 - 1 * x1 * x2 ** 2
    Delta_b = x0 ** 2 * y1 * 1 + y0 * 1 * x2 ** 2 + 1 * x1 ** 2 * y2 - x0 ** 2 * 1 * y2 - y0 * x1 ** 2 * 1 - 1 * y1 * x2 ** 2
    Delta_c = x0 ** 2 * x1 * y2 + x0 * y1 * x2 ** 2 + y0 * x1 ** 2 * x2 - x0 ** 2 * y1 * x2 - x0 * x1 ** 2 * y2 - y0 * x1 * x2 ** 2
    a = Delta_a / DetA
    b = Delta_b / DetA
    c = Delta_c / DetA

    x_2dArr = np.vstack((x0, x1, x2))
    x_start = np.min(x_2dArr, axis=0)
    x_end = np.max(x_2dArr, axis=0)
    x_localmin = -b / (2 * a)
    y_localmin = a * x_localmin ** 2 + b * x_localmin + c
    flag_x_within_interval = (x_start <= x_localmin) & (x_localmin <= x_end) & (a >= 0)
    # flag_x_within_interval = (a >= 0)

    min_x, min_y = np.zeros_like(x0).astype('float'), np.zeros_like(y0).astype('float')
    min_x[flag_x_within_interval] = x_localmin[flag_x_within_interval]
    min_y[flag_x_within_interval] = y_localmin[flag_x_within_interval]
    min_x[~flag_x_within_interval] = x_start[~flag_x_within_interval]
    min_y[~flag_x_within_interval] = y_localmin[~flag_x_within_interval] * float(math.inf)

    restored_a, restored_b, restored_c = \
        restore_shape(a, x0_raw.shape), restore_shape(b, x0_raw.shape), restore_shape(c, x0_raw.shape)
    restored_min_x, restored_min_y = restore_shape(min_x, x0_raw.shape), restore_shape(min_y, x0_raw.shape)
    restored_flag_x_within_interval = restore_shape(flag_x_within_interval, x0_raw.shape)

    return restored_a, restored_b, restored_c, restored_min_x, restored_min_y, \
           restored_flag_x_within_interval

def find_segment_min():
    pass


# 向量化操作函数
def find_min_with_vectorization(arr):
    min_indices = np.argmin(arr, axis=2)

    arr_X_len, arr_Y_len, arr_Z_len = arr.shape[0], arr.shape[1], arr.shape[2]
    min_index_3points = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=int)
    index_Y_2D, index_X_2D = np.meshgrid(np.arange(arr_Y_len), np.arange(arr_X_len))

    min_index_3points[:, :, 0] = min_indices - 1
    min_index_3points[:, :, 1] = min_indices
    min_index_3points[:, :, 2] = min_indices + 1

    flag_left, flag_right = min_index_3points[:, :, 0] < 0, min_index_3points[:, :, 2] > arr_Z_len - 1
    min_index_3points[flag_left, :] += 1
    min_index_3points[flag_right, :] -= 1
    min_values_3points_0 = arr[index_X_2D, index_Y_2D, min_index_3points[:, :, 0]]
    min_values_3points_1 = arr[index_X_2D, index_Y_2D, min_index_3points[:, :, 1]]
    min_values_3points_2 = arr[index_X_2D, index_Y_2D, min_index_3points[:, :, 2]]
    min_values_3points = \
        np.dstack((min_values_3points_0, min_values_3points_1, min_values_3points_2))

    return min_index_3points, min_values_3points

def GetMatrixDerv(X, Matrix, RowFlag=True):
    # If RowFlag is True, calculate derivatives for each row of Matrix.
    # Otherwise, calculate derivatives for each column.
    MatrixRowNum, MatrixColNum = np.size(Matrix, 0), np.size(Matrix, 1)
    if RowFlag:
        XExtend = np.tile(X, (MatrixRowNum, 1))
        dMdX = np.gradient(Matrix, axis=1) / np.gradient(XExtend, axis=1)
    else:
        XExtend = np.tile(X, (MatrixColNum, 1)).T
        dMdX = np.gradient(Matrix, axis=0) / np.gradient(XExtend, axis=0)
    return dMdX

def MyMatrixExtend(arr, m, RowFlag=True):
    # n = np.size(arr)
    if RowFlag:
        arr_new = np.tile(arr, (m, 1))
    else:
        arr_new = np.tile(arr, (m, 1)).T
    return arr_new

if __name__ == "__main__":
    # load the spiral line
    SpiralData = np.loadtxt('test.spiral', skiprows=1)
    curve_f, curve_r = SpiralData[:, 0], SpiralData[:, 1]
    curve_x, curve_y = curve_r * np.cos(curve_f), curve_r * np.sin(curve_f)

    # 生成50x50的网格点坐标
    # x_min, x_max, n_grid_x = -7, 7, 1000
    # y_min, y_max, n_grid_y = -7, 7, 1000
    # Xmesh, Ymesh = np.meshgrid(np.linspace(x_min, x_max, n_grid_x), np.linspace(y_min, y_max, n_grid_y))

    fi_min, fi_max, n_fi_grid = 0, 360, 1081
    r_min, r_max, n_grid_r = 3.0, 7.0, 1001
    fi_axis, r_axis = np.linspace(fi_min, fi_max, n_fi_grid), np.linspace(r_min, r_max, n_grid_r)
    Fmesh, Rmesh = np.meshgrid(fi_axis, r_axis)
    Xmesh, Ymesh = Rmesh*np.cos(Fmesh/180.0*np.pi), Rmesh*np.sin(Fmesh/180.0*np.pi)

    # 向量化计算网格点到曲线上所有点的距离
    start_time_vectorized = time.time()
    distances_vectorized = np.sqrt((Xmesh[..., np.newaxis] - curve_x) ** 2 + (Ymesh[..., np.newaxis] - curve_y) ** 2)
    end_time_vectorized = time.time()
    time_vectorized = end_time_vectorized - start_time_vectorized

    # 遍历每个grid, 找到最小值附近的3个点及相应的索引
    min_index_3points, min_values_3points = find_min_with_vectorization(distances_vectorized)

    # 进行二次函数拟合,找到较精确的极小值，遍历每个grid,返回距离极小值,极小值对应的曲线上的点
    f0_raw, d0_raw = curve_f[min_index_3points[:, :, 0]], min_values_3points[:, :, 0]
    f1_raw, d1_raw = curve_f[min_index_3points[:, :, 1]], min_values_3points[:, :, 1]
    f2_raw, d2_raw = curve_f[min_index_3points[:, :, 2]], min_values_3points[:, :, 2]

    restored_a, restored_b, restored_c, restored_min_f, restored_min_dist, \
    restored_flag_x_within_interval = find_quadratic_min(f0_raw, d0_raw, f1_raw, d1_raw, f2_raw, d2_raw)

    #获取spiral最小距离点到网格点的方向向量
    _, FuncSpiral_derv, FuncSpiral = linear_interpolation_vectorization(curve_f, curve_r)
    restored_min_r = FuncSpiral(restored_min_f)
    restored_min_rp = FuncSpiral_derv(restored_min_f)
    restored_min_x, restored_min_y = \
        restored_min_r * np.cos(restored_min_f), restored_min_r * np.sin(restored_min_f)
    # directions
    tang_direction_x, tang_direction_y = \
        restored_min_rp*np.cos(restored_min_f)-restored_min_r*np.sin(restored_min_f),\
        restored_min_rp*np.sin(restored_min_f)+restored_min_r*np.cos(restored_min_f)
    RadialDirection_x, RadialDirection_y = Xmesh, Ymesh
    AzimuthDirection_x,  AzimuthDirection_y = -Ymesh, Xmesh
    # Length
    Length_E_vect = np.sqrt(tang_direction_x**2+tang_direction_y**2)
    Length_xyPos_vect = np.sqrt(RadialDirection_x**2+RadialDirection_y**2)

    #spiral指向网格点的向量
    tang_direction_x_norm, tang_direction_y_norm = tang_direction_x/Length_E_vect, tang_direction_y/Length_E_vect
    norm_direction_x_norm, norm_direction_y_norm = -tang_direction_y_norm, tang_direction_x_norm
    #网格点的径向向量
    RadialDirection_x_norm, RadialDirection_y_norm = \
        RadialDirection_x/Length_xyPos_vect, RadialDirection_y/Length_xyPos_vect
    #网格点的切向向量
    AzimuthDirection_x_norm, AzimuthDirection_y_norm = \
        AzimuthDirection_x/Length_xyPos_vect, AzimuthDirection_y/Length_xyPos_vect

    FWHM, U = 0.60, 1
    delta = FWHM / 2.35482

    EhMesh = U / delta / np.sqrt(2 * np.pi) * np.exp(-0.5 * (restored_min_dist / delta) ** 2)
    EfiMesh = EhMesh * (norm_direction_x_norm*AzimuthDirection_x_norm+norm_direction_y_norm*AzimuthDirection_y_norm)
    ErMesh = EhMesh * (norm_direction_x_norm*RadialDirection_x_norm+norm_direction_y_norm*RadialDirection_y_norm)

    # fi_axis, r_axis
    dErdR = GetMatrixDerv(r_axis, ErMesh, RowFlag=False)#dEr/dr
    dEfdF = GetMatrixDerv(fi_axis, EfiMesh, RowFlag=True)#dEfi/dfi
    rMatrix=MyMatrixExtend(r_axis,np.size(fi_axis),RowFlag=False)
    EzMesh = -1/rMatrix*(ErMesh+rMatrix*dErdR+dEfdF)

    # 创建3D图形对象
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    # 绘制3D曲面
    ax.plot_surface(Xmesh, Ymesh, EhMesh, cmap='cool', rcount=120, ccount=120)
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 创建3D图形对象
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    # 绘制3D曲面
    ax.plot_surface(Xmesh, Ymesh, EfiMesh, cmap='cool', rcount=120, ccount=120)
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 创建3D图形对象
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    # 绘制3D曲面
    ax.plot_surface(Xmesh, Ymesh, ErMesh, cmap='cool', rcount=120, ccount=120)
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 创建3D图形对象
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    # 绘制3D曲面
    ax.plot_surface(Xmesh, Ymesh, dEfdF, cmap='cool', rcount=120, ccount=120)
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    np.savetxt('Xmesh_E_plot.txt', Xmesh)
    np.savetxt('Ymesh_E_plot.txt', Ymesh)
    np.savetxt('EhMesh_E_plot.txt', EhMesh)
    np.savetxt('EfMesh_E_plot.txt', EfiMesh)
    np.savetxt('Ermesh_E_plot.txt', ErMesh)
    # 显示图形
    plt.show()
