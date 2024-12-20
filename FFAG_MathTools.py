import numpy as np
import math
import copy
from numba import njit, types
from numba.np.extensions import cross2d
from mpi4py import MPI


class FFAG_interpolation:
    def __init__(self):
        pass

    def find_two_points(self, X, xi):
        """
        Find the nearest two sample points around xi and return their positions
        Input: X: an array of sample point positions (sorted in ascending order)
               xi: the target value
        Output: left_point, right_point: two positions of the closest points
                flag: a flag indicating whether xi lies within the range of X (1) or not (0)
                Note: left_point <= xi <= right_point
        """
        delta_x = X[1] - X[0]
        idx = math.floor((xi - X[0]) / delta_x)
        if idx < 0:
            flag = 1
            idx_array = np.array((0, 1))
        elif idx >= len(X) - 2:
            flag = 1
            idx_array = np.array((len(X) - 2, len(X) - 1))
        else:
            flag = 0
            idx_array = np.array((idx, idx + 1))
        return idx_array, flag

    def find_two_points_non_uniform(self, X, xi):
        """
        Find the nearest two sample points around xi and return their positions
        Input: X: an array of sample point positions (sorted in ascending order)
               xi: the target value
        Output: left_point, right_point: two positions of the closest points
                flag: a flag indicating whether xi lies within the range of X (1) or not (0)
                Note: left_point <= xi <= right_point
        """
        right_point = np.searchsorted(X, xi, side='right')
        left_point = right_point - 1
        # Check if xi is out of the range of X
        flag = 0
        if left_point < 0:
            idx_array = np.array((0, 1))
            flag = 1
        elif right_point > len(X) - 1:
            idx_array = np.array((len(X) - 2, len(X) - 1))
            flag = 1
        else:
            idx_array = np.array([left_point, right_point])

        return idx_array, flag

    def find_two_points_non_uniform_vect(self, X, xi):

        right_point = np.searchsorted(X, xi, side='right')
        left_point = right_point - 1

        # Check if xi is out of the range of X
        idx_array_0, idx_array_1, flag = (
            copy.deepcopy(left_point), copy.deepcopy(right_point), np.zeros_like(xi, dtype=bool))

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

    def find_four_points_non_uniform(self, X, xi):
        """
        Find the nearest four sample points around xi and return their positions
        Input: X: an array of sample point positions (sorted in ascending order)
               xi: the target value
        Output: left_point, right_point: two positions of the closest points
                flag: a flag indicating whether xi lies within the range of X (1) or not (0)
                Note: left_point <= xi <= right_point
        """
        right_point = np.searchsorted(X, xi, side='right')
        left_point = right_point - 1
        right_right_point = right_point + 1
        left_left_point = left_point - 1
        # Check if xi is out of the range of X
        flag = 0
        if left_left_point < 0:
            idx_array = np.array((0, 1, 2, 3))
            flag = 1
        elif right_right_point > len(X) - 1:
            idx_array = np.array((len(X) - 4, len(X) - 3, len(X) - 2, len(X) - 1))
            flag = 1
        else:
            idx_array = np.array((left_left_point, left_point, right_point, right_right_point))
        return idx_array, flag

    def find_four_points_non_uniform_vector(self, X, xi):
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
                np.zeros_like(xi, dtype=bool)

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

        flag_left_point_equal_0 = left_point == 0
        flag_left_point_equal_minus2 = left_point == len(X) - 2
        flag[flag_left_point_equal_0] = 0
        flag[flag_left_point_equal_minus2] = 0

        idx_array = np.vstack((idx_array_0, idx_array_1, idx_array_2, idx_array_3))

        return idx_array, flag

    def linear_interpolation(self, X, Y, NonUniform=True):
        def interpolator(xi):
            if NonUniform:
                idx_array, OutRangeFlag = self.find_two_points_non_uniform(X, xi)
            else:
                idx_array, OutRangeFlag = self.find_two_points(X, xi)
            x = X[idx_array]
            y = Y[idx_array]
            x0, x1, y0, y1 = x[0], x[1], y[0], y[1]
            return y0 + (xi - x0) / (x1 - x0) * (y1 - y0)

        return interpolator

    def linear_interpolation_vect(self, X, Y, NonUniform=True):
        def interpolator(xi):
            if NonUniform:
                idx_array, OutRangeFlag = self.find_two_points_non_uniform_vect(X, xi)
            else:
                idx_array, OutRangeFlag = self.find_two_points(X, xi)
            x = X[idx_array]
            y = Y[idx_array]
            x0, x1, y0, y1 = x[0], x[1], y[0], y[1]
            return y0 + (xi - x0) / (x1 - x0) * (y1 - y0)

        return interpolator

    def derivate_interpolation(self, X, Y):
        def derivate_interpolator(xi):
            idx_array, OutRangeFlag = self.find_four_points_non_uniform_vector(X, xi)
            x = X[idx_array]
            y = Y[idx_array]
            coefficients = np.polyfit(x, y, deg=3)
            a3, a2, a1, a0 = coefficients[0], coefficients[1], coefficients[2], coefficients[3]
            return 3 * a3 * xi ** 2 + 2 * a2 * xi + a1

        return derivate_interpolator

    def find_four_points(self, X, xi):
        """
        Find the four sample points around xi and return them
        Input: X: array of sample point positions
               xi: the interpolation point
        Output: arrays of X positions and values of four closest sample points
                flag: an indicator in case xi falls outside range of X values (1) or not (0)
        """
        delta_x = X[1] - X[0]
        idx = math.floor((xi - X[0]) / delta_x)
        if idx < 1:
            flag = 1
            idx_array = np.array((0, 1, 2, 3))
        elif idx >= len(X) - 2:
            flag = 1
            # idx = len(X) - 2
            idx_array = np.array((len(X) - 2 - 2, len(X) - 2 - 1, len(X) - 2, len(X) - 2 + 1))
        else:
            flag = 0
            idx_array = np.array((idx - 1, idx, idx + 1, idx + 2))

        if idx == 0 or idx == len(X) - 2:
            flag = 0

        return idx_array, flag

    def polynomial_interpolation(self, X, Y, xi):
        """
        Polynomial interpolation function
        Input: X, Y: arrays of sample point positions and values
               xi: the interpolation point
        Output: yi: interpolated value at xi
                OutRangeFlag: an indicator in case xi falls outside range of X values (1) or not (0)
        """
        idx_array, OutRangeFlag = self.find_four_points(X, xi)
        x = X[idx_array]
        y = Y[idx_array]
        x_i_0, x_i_1, x_i_2, x_i_3 = xi - x[0], xi - x[1], xi - x[2], xi - x[3]
        x_0_1, x_0_2, x_0_3 = x[0] - x[1], x[0] - x[2], x[0] - x[3]
        x_1_2, x_1_3 = x[1] - x[2], x[1] - x[3]
        x_2_3 = x[2] - x[3]
        # L0 = y[0] * (xi - x[1]) * (xi - x[2]) * (xi - x[3]) / (x[0] - x[1]) / (x[0] - x[2]) / (x[0] - x[3])
        # L1 = y[1] * (xi - x[0]) * (xi - x[2]) * (xi - x[3]) / (x[1] - x[0]) / (x[1] - x[2]) / (x[1] - x[3])
        # L2 = y[2] * (xi - x[0]) * (xi - x[1]) * (xi - x[3]) / (x[2] - x[0]) / (x[2] - x[1]) / (x[2] - x[3])
        # L3 = y[3] * (xi - x[0]) * (xi - x[1]) * (xi - x[2]) / (x[3] - x[0]) / (x[3] - x[1]) / (x[3] - x[2])
        L0 = y[0] * x_i_1 * x_i_2 * x_i_3 / x_0_1 / x_0_2 / x_0_3
        L1 = -1 * y[1] * x_i_0 * x_i_2 * x_i_3 / x_0_1 / x_1_2 / x_1_3
        L2 = y[2] * x_i_0 * x_i_1 * x_i_3 / x_0_2 / x_1_2 / x_2_3
        L3 = -1 * y[3] * x_i_0 * x_i_1 * x_i_2 / x_0_3 / x_1_3 / x_2_3
        Value = L0 + L1 + L2 + L3
        return Value, OutRangeFlag

    def polynomial_interpolation_2D(self, X, Y, Z, Xi, Yi):
        """
        2D polynomial interpolation function
        Input: X, Y: arrays of sample point positions in the vertical(r_axis) and horizontal(fi_axis) directions, respectively
               Z: 2D array of function values at the sample points
               Xi, Yi: coordinates of the interpolation point to be evaluated
        Output: Value: interpolated function value at the point (Xi, Yi)
                OutRangeFlag: an indicator in case (Xi, Yi) falls outside the range of sample point positions
                (1) or not (0)
        """
        # First, use the find_four_points function to find the positions of the four closest sample points in the
        # vertical direction around the given interpolation point with horizontal position xi
        idx_y, flag = self.find_four_points(Y, Yi)

        # Use polynomial_interpolation function to perform 1D interpolation on the four closest sample points in the
        # horizontal direction around the given interpolation point with vertical position Yi, and obtain the values of these
        # four interpolated points
        SampleY0, SampleY1, SampleY2, SampleY3 = Z[:, idx_y[0]], Z[:, idx_y[1]], Z[:, idx_y[2]], Z[:, idx_y[3]]
        SampleX0, OutRangeFlag0 = self.polynomial_interpolation(X, SampleY0, Xi)
        SampleX1, OutRangeFlag1 = self.polynomial_interpolation(X, SampleY1, Xi)
        SampleX2, OutRangeFlag2 = self.polynomial_interpolation(X, SampleY2, Xi)
        SampleX3, OutRangeFlag3 = self.polynomial_interpolation(X, SampleY3, Xi)

        # Combine the values of the four interpolated points into a new array SampleX
        SampleX = np.array([SampleX0, SampleX1, SampleX2, SampleX3])

        # Use polynomial_interpolation function to perform 1D interpolation on the four closest sample points in the
        # vertical direction around the given interpolation point with horizontal position xi, using SampleX as the values.
        # Obtain the value of the final interpolation point
        Value, _ = self.polynomial_interpolation(
            np.array([Y[idx_y[0]], Y[idx_y[1]], Y[idx_y[2]], Y[idx_y[3]]]), SampleX, Yi)
        OutRangeFlag = flag + OutRangeFlag0 + OutRangeFlag1 + OutRangeFlag2 + OutRangeFlag3
        # Return the value of the final interpolation point and a flag indicating whether the interpolation point is within
        # the range of sample point positions or not
        return Value, OutRangeFlag

    def Lagrange_interp_2D_vect(self, X, Y, Z, Xi, Yi):
        """
        Input: X, Y: arrays of sample point positions in the vertical(r_axis) and horizontal(fi_axis) directions, respectively
               Z: 2D array of function values at the sample points
               Xi, Yi: coordinates of the interpolation point to be evaluated
        """
        idx_array_x, flag_x = self.find_four_points_non_uniform_vector(X, Xi)
        idx_array_y, flag_y = self.find_four_points_non_uniform_vector(Y, Yi)
        OutRangeFlag = np.logical_or(flag_x, flag_y)
        # shape of idx_array_x, idx_array_y: 4 * nSamplePoints
        nSamplePoints = np.size(Xi, 0)

        xs, ys = X[idx_array_x], Y[idx_array_y]  # x and y coordinates of the surrounding points
        idx_grid_x, idx_grid_y = self.My3DMeshgrid(idx_array_x, idx_array_y)
        zs = Z[idx_grid_x, idx_grid_y]

        Lx, Ly, Lxy = np.zeros((4, nSamplePoints)), np.zeros((4, nSamplePoints)), np.zeros((4, 4, nSamplePoints))

        Lx[0, :] = (Xi - xs[1, :]) * (Xi - xs[2, :]) * (Xi - xs[3, :]) / (xs[0, :] - xs[1, :]) / (
                xs[0, :] - xs[2, :]) / (xs[0, :] - xs[3, :])
        Lx[1, :] = (Xi - xs[0, :]) * (Xi - xs[2, :]) * (Xi - xs[3, :]) / (xs[1, :] - xs[0, :]) / (
                xs[1, :] - xs[2, :]) / (xs[1, :] - xs[3, :])
        Lx[2, :] = (Xi - xs[0, :]) * (Xi - xs[1, :]) * (Xi - xs[3, :]) / (xs[2, :] - xs[0, :]) / (
                xs[2, :] - xs[1, :]) / (xs[2, :] - xs[3, :])
        Lx[3, :] = (Xi - xs[0, :]) * (Xi - xs[1, :]) * (Xi - xs[2, :]) / (xs[3, :] - xs[0, :]) / (
                xs[3, :] - xs[1, :]) / (xs[3, :] - xs[2, :])

        Ly[0, :] = (Yi - ys[1, :]) * (Yi - ys[2, :]) * (Yi - ys[3, :]) / (ys[0, :] - ys[1, :]) / (
                ys[0, :] - ys[2, :]) / (ys[0, :] - ys[3, :])
        Ly[1, :] = (Yi - ys[0, :]) * (Yi - ys[2, :]) * (Yi - ys[3, :]) / (ys[1, :] - ys[0, :]) / (
                ys[1, :] - ys[2, :]) / (ys[1, :] - ys[3, :])
        Ly[2, :] = (Yi - ys[0, :]) * (Yi - ys[1, :]) * (Yi - ys[3, :]) / (ys[2, :] - ys[0, :]) / (
                ys[2, :] - ys[1, :]) / (ys[2, :] - ys[3, :])
        Ly[3, :] = (Yi - ys[0, :]) * (Yi - ys[1, :]) * (Yi - ys[2, :]) / (ys[3, :] - ys[0, :]) / (
                ys[3, :] - ys[1, :]) / (ys[3, :] - ys[2, :])

        for idx_Lx in range(0, 4):
            for idx_Ly in range(0, 4):
                Lxy[idx_Lx, idx_Ly, :] = Lx[idx_Lx, :] * Ly[idx_Ly, :]

        interp_value = np.sum(Lxy * zs, axis=(0, 1))
        return interp_value, OutRangeFlag

    def My3DMeshgrid(self, x_2D, y_2D):
        # shape of x_2D and y_2D: D*N, N=number of sample points
        # D = dimension of the coordinate of the sample points
        DimensionSamplePoints = np.size(x_2D, 0)
        x_expand, y_expand = x_2D[:, np.newaxis, :], y_2D[np.newaxis, :, :]
        # Create a meshgrid for x
        X = np.tile(x_expand, (1, DimensionSamplePoints, 1))
        # Create a meshgrid for y
        Y = np.tile(y_expand, (DimensionSamplePoints, 1, 1))
        # X[:,:,0] is same in the row, Y[:,:,0] is same in the column
        return X, Y

    def My2p5DInterp(self, Matrix3D, axis_dim0, axis_dim1):
        # axis0: steps of azimuth
        # axis1: PID
        # axis2: coordinates
        # 在axis0, axis1上进行插值
        LenDim3 = np.size(Matrix3D, 2)

        def InterpFunc2D(fi0, PID_n):
            idx_array_x, flag_x = self.find_two_points_non_uniform_vect(axis_dim0, fi0)
            idx_array_y, flag_y = self.find_two_points_non_uniform_vect(axis_dim1, PID_n)
            OutRangeFlag = np.logical_or(flag_x, flag_y)
            idx_x0, idx_x1 = idx_array_x[0, :], idx_array_x[1, :]
            idx_y0, idx_y1 = idx_array_y[0, :], idx_array_y[1, :]

            x0, x1 = axis_dim0[idx_x0], axis_dim0[idx_x1]
            y0, y1 = axis_dim1[idx_y0], axis_dim1[idx_y1]

            z00 = Matrix3D[idx_x0, idx_y0, :]  # at x0,y0
            z01 = Matrix3D[idx_x0, idx_y1, :]  # at x0,y1
            z10 = Matrix3D[idx_x1, idx_y0, :]  # at x1,y0
            z11 = Matrix3D[idx_x1, idx_y1, :]  # at x1,y1
            L00 = (fi0 - x1) * (PID_n - y1) / (x0 - x1) / (y0 - y1)
            L01 = (fi0 - x1) * (PID_n - y0) / (x0 - x1) / (y1 - y0)
            L10 = (fi0 - x0) * (PID_n - y1) / (x1 - x0) / (y0 - y1)
            L11 = (fi0 - x0) * (PID_n - y0) / (x1 - x0) / (y1 - y0)
            L00_Ext = np.tile(L00, (LenDim3, 1)).T
            L01_Ext = np.tile(L01, (LenDim3, 1)).T
            L10_Ext = np.tile(L10, (LenDim3, 1)).T
            L11_Ext = np.tile(L11, (LenDim3, 1)).T

            # fi0 equals to x0, PID_n equals to y
            result = L00_Ext * z00 + L01_Ext * z01 + L10_Ext * z10 + L11_Ext * z11

            return result, OutRangeFlag

        return InterpFunc2D

    def My1p5DInterp(self, X_values, Y_arrs):
        # x是一维数组，y是二维矩阵，x中的每一个数对应y的一行
        def Interp1p5D(xi):
            idx_array, flag = self.find_two_points_non_uniform_vect(X_values, xi)
            x0, x1 = X_values[idx_array[0]], X_values[idx_array[1]]
            y0, y1 = Y_arrs[idx_array[0], :], Y_arrs[idx_array[1], :]
            y = (xi-x1)/(x0-x1)*y0 + (xi-x0)/(x1-x0)*y1
            return y, flag

        return Interp1p5D


class FFAG_derivative:
    def __init__(self):
        pass

    def myderivative(self, x, y):
        # First-order derivative in difference form
        # test code:
        # Declare an array deriv of zeros of length n
        n = len(x)
        deriv = np.zeros(n)

        # Calculate first derivative at interior points
        # using central divided differences
        for i in range(1, n - 1):
            deriv[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        # Calculate first derivative at end points using
        # forward and backward divided differences
        deriv[0] = (y[1] - y[0]) / (x[1] - x[0])
        deriv[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])

        return deriv


    def myderivative_smooth(self, x, y):
        # Declare an array deriv of zeros of length n
        n = len(x)
        deriv = np.zeros(n)

        # Calculate first derivative at interior points
        # using central divided differences
        for i in range(1, n - 1):
            deriv[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        # Calculate first derivative at end points using
        # forward and backward divided differences
        deriv[0] = 2 * deriv[1] - deriv[2]
        deriv[n - 1] = 2 * deriv[n - 2] - deriv[n - 3]

        return deriv

    def mysecondderivative(self, x, y):
        # Second-order derivative in difference form
        n = len(x)
        deriv2 = np.zeros(n)
        deriv2_interp_on_boundary = np.zeros(n)

        for i in range(1, n - 1):
            h1 = x[i] - x[i - 1]
            h2 = x[i + 1] - x[i]
            h_sum = h1 + h2

            deriv2[i] = 2 * ((y[i + 1] - y[i]) / (h2 * h_sum) - (y[i] - y[i - 1]) / (h1 * h_sum))
            deriv2_interp_on_boundary[i] = deriv2[i]

        h0 = x[1] - x[0]
        h1 = x[2] - x[1]
        h_sum = h0 + h1
        deriv2[0] = 2 * ((y[2] - y[1]) / (h1 * h_sum) - (y[1] - y[0]) / (h0 * h_sum))
        deriv2_interp_on_boundary[0] = 2 * deriv2[1] - deriv2[2]

        hn_minus_2 = x[n - 2] - x[n - 3]
        hn_minus_1 = x[n - 1] - x[n - 2]
        h_sum = hn_minus_2 + hn_minus_1
        deriv2[n - 1] = 2 * (
                (y[n - 1] - y[n - 2]) / (hn_minus_1 * h_sum) - (y[n - 2] - y[n - 3]) / (hn_minus_2 * h_sum))
        deriv2_interp_on_boundary[n - 1] = 2 * deriv2[n - 2] - deriv2[n - 3]

        return deriv2_interp_on_boundary, deriv2

    def MyMatrixExtend(self, arr, m, RowFlag=True):
        # n = np.size(arr)
        if RowFlag:
            arr_new = np.tile(arr, (m, 1))
        else:
            arr_new = np.tile(arr, (m, 1)).T
        return arr_new

    def myderv_matrix(self, x, matrixY, RowFlag=True):
        # First-order derivative in matrix form
        rownum, colnum = np.size(matrixY, 0), np.size(matrixY, 1)
        dMdX = np.zeros_like(matrixY)
        if RowFlag:
            # matrixX = np.tile(x, (rownum, 1))
            matrixX = self.MyMatrixExtend(x, rownum, RowFlag)
            dMdX[:, 1:-1] = (matrixY[:, 2:] - matrixY[:, 0:-2]) / (matrixX[:, 2:] - matrixX[:, 0:-2])
            # At the start point and the end point, using interpolation
            dMdX[:, 0] = 2 * dMdX[:, 1] - dMdX[:, 2]
            dMdX[:, -1] = 2 * dMdX[:, -2] - dMdX[:, -3]
        else:
            # matrixX = np.tile(x, (rownum, 1)).T
            matrixX = self.MyMatrixExtend(x, colnum, RowFlag)
            dMdX[1:-1, :] = (matrixY[2:, :] - matrixY[0:-2, :]) / (matrixX[2:, :] - matrixX[0:-2, :])
            dMdX[0, :] = 2 * dMdX[1, :] - dMdX[2, :]
            dMdX[-1, :] = 2 * dMdX[-2, :] - dMdX[-3, :]

        return dMdX


class FFAG_Algorithm:
    def __init__(self):
        pass

    def BiSection(self, x_left, x_right, func, paras=None):
        x_mid, y_mid = float("nan"), float("nan")
        for k in range(0, 100):
            x_mid = (x_left + x_right) / 2
            y_left, y_right, y_mid = func(x_left, paras), func(x_right, paras), func(x_mid, paras)
            if y_mid * y_left < 0:
                x_right = x_mid
            else:
                x_left = x_mid
            # print("ymid=",y_mid)
        return x_mid, y_mid

    def nelder_mead(self, obj_func, initial_simplex, ObjFuncPara, max_iter=100, alpha=1.0, gamma=2.0, rho=0.5,
                    eps=1e-5):
        n = initial_simplex.shape[1]  # dimension
        func_values = np.zeros(n + 1)  # target function value
        vertex_flag = np.zeros(n + 1)  # if flag = 0, the corresponding vertex should be updated
        n_index = np.arange(0, n+1)
        # n_indexLocal = FFAG_MPI().DivideVariables(n_index)
        for i in n_index:
            func_values[i] = obj_func(initial_simplex[i], ObjFuncPara)
            vertex_flag[i] = 1

        for j in range(max_iter):
            indices = np.argsort(func_values)  # sort
            best, second_best, worst = indices[0], indices[1], indices[-1]
            # if rank==0:
            print("j=", j, "best=", func_values[best])
            print("j=", j, "worst=", func_values[worst])
            if func_values[best] < 1:
                break
            best_x, second_best_x, worst_x = initial_simplex[best], initial_simplex[second_best], initial_simplex[worst]
            best_FuncValue, second_best_FuncValue, worst_FuncValue = func_values[best], func_values[second_best], \
                func_values[worst]
            center = np.mean(initial_simplex[indices[:-1]], axis=0)
            # reflect
            reflect_x = center + alpha * (center - worst_x)
            reflect_FuncValue = obj_func(reflect_x, ObjFuncPara)
            if reflect_FuncValue < second_best_FuncValue and \
                    reflect_FuncValue >= best_FuncValue:
                initial_simplex[worst] = reflect_x
                # print('reflect')
                vertex_flag[worst] = 0
            # expand
            elif reflect_FuncValue < best_FuncValue:
                expand_x = center + gamma * (reflect_x - center)
                expand_FuncValue = obj_func(expand_x, ObjFuncPara)
                if expand_FuncValue < best_FuncValue:
                    initial_simplex[worst] = expand_x
                    # print('expand')
                    vertex_flag[worst] = 0
                else:
                    initial_simplex[worst] = reflect_x
                    # print('reflect')
                    vertex_flag[worst] = 0
            # contract
            else:
                contract_x = center + rho * (worst_x - center)
                contract_FuncValue = obj_func(contract_x, ObjFuncPara)
                if contract_FuncValue < worst_FuncValue:
                    initial_simplex[worst] = contract_x
                    # print('contract1')
                    vertex_flag[worst] = 0
                else:  # contract
                    initial_simplex[indices[1:]] = center + (initial_simplex[indices[1:]] - center) / 2
                    # print('contract2')
                    vertex_flag[indices[1:]] = 0
            # calculate eps
            center = np.mean(initial_simplex, axis=0)
            range_ = np.max(np.abs(initial_simplex - center))
            # if range_ < eps:
            #     break
            for i in range(n + 1):
                if vertex_flag[i] == 0:
                    func_values[i] = obj_func(initial_simplex[i], ObjFuncPara)

        index_min = np.argmin(func_values)
        return initial_simplex[index_min]


class FFAG_ANN():
    def __init__(self):
        pass

class FFAG_BasicTool:
    def __init__(self):
        pass

    def mod2zero(self, x, y):
        mod_value = x - (x/y).astype(int)*y
        return mod_value


# numba optimized functions
@njit
def find_four_points_non_uniform_vector(X, xi):
    right_point = np.searchsorted(X, xi, side='right')
    left_point = right_point - 1
    right_right_point = right_point + 1
    left_left_point = left_point - 1

    flag_left_point_less_than_0 = left_left_point < 0
    flag_right_point_larger_than_len = right_right_point > len(X) - 1
    idx_array_0, idx_array_1, idx_array_2, idx_array_3, flag = \
        left_left_point, left_point, right_point, right_right_point, \
        np.zeros_like(xi)

    idx_array_0[flag_left_point_less_than_0] = 0
    idx_array_1[flag_left_point_less_than_0] = 1
    idx_array_2[flag_left_point_less_than_0] = 2
    idx_array_3[flag_left_point_less_than_0] = 3
    flag[flag_left_point_less_than_0] = 1

    idx_array_0[flag_right_point_larger_than_len] = len(X) - 4
    idx_array_1[flag_right_point_larger_than_len] = len(X) - 3
    idx_array_2[flag_right_point_larger_than_len] = len(X) - 2
    idx_array_3[flag_right_point_larger_than_len] = len(X) - 1
    flag[flag_right_point_larger_than_len] = 1

    flag_left_point_equal_0 = left_point == 0
    flag_left_point_equal_minus2 = left_point == len(X) - 2
    flag[flag_left_point_equal_0] = 0
    flag[flag_left_point_equal_minus2] = 0

    idx_array = np.vstack((idx_array_0, idx_array_1, idx_array_2, idx_array_3))

    return idx_array, flag


@njit
def Interp_DoSum(xs, ys, zs, Xi, Yi):

    Lx = np.zeros((4, len(Xi)))
    Ly = np.zeros((4, len(Yi)))
    Lxy = np.zeros((4, 4, len(Xi)))

    Lx[0, :] = (Xi - xs[1, :]) * (Xi - xs[2, :]) * (Xi - xs[3, :]) / (xs[0, :] - xs[1, :]) / (
            xs[0, :] - xs[2, :]) / (xs[0, :] - xs[3, :])
    Lx[1, :] = (Xi - xs[0, :]) * (Xi - xs[2, :]) * (Xi - xs[3, :]) / (xs[1, :] - xs[0, :]) / (
            xs[1, :] - xs[2, :]) / (xs[1, :] - xs[3, :])
    Lx[2, :] = (Xi - xs[0, :]) * (Xi - xs[1, :]) * (Xi - xs[3, :]) / (xs[2, :] - xs[0, :]) / (
            xs[2, :] - xs[1, :]) / (xs[2, :] - xs[3, :])
    Lx[3, :] = (Xi - xs[0, :]) * (Xi - xs[1, :]) * (Xi - xs[2, :]) / (xs[3, :] - xs[0, :]) / (
            xs[3, :] - xs[1, :]) / (xs[3, :] - xs[2, :])

    Ly[0, :] = (Yi - ys[1, :]) * (Yi - ys[2, :]) * (Yi - ys[3, :]) / (ys[0, :] - ys[1, :]) / (
            ys[0, :] - ys[2, :]) / (ys[0, :] - ys[3, :])
    Ly[1, :] = (Yi - ys[0, :]) * (Yi - ys[2, :]) * (Yi - ys[3, :]) / (ys[1, :] - ys[0, :]) / (
            ys[1, :] - ys[2, :]) / (ys[1, :] - ys[3, :])
    Ly[2, :] = (Yi - ys[0, :]) * (Yi - ys[1, :]) * (Yi - ys[3, :]) / (ys[2, :] - ys[0, :]) / (
            ys[2, :] - ys[1, :]) / (ys[2, :] - ys[3, :])
    Ly[3, :] = (Yi - ys[0, :]) * (Yi - ys[1, :]) * (Yi - ys[2, :]) / (ys[3, :] - ys[0, :]) / (
            ys[3, :] - ys[1, :]) / (ys[3, :] - ys[2, :])

    for i in range(4):
        for j in range(4):
            Lxy[i, j, :] = Lx[i, :] * Ly[j, :]

    return Lxy * zs

@njit
def Interp_DoSum_new(xs, ys, zs, Xi, Yi):
    Lx = np.zeros((4, len(Xi)))
    Ly = np.zeros((4, len(Yi)))
    Lxy = np.zeros((4, 4, len(Xi)))

    # 预计算 xs 的差值
    xs_01 = xs[0, :] - xs[1, :]
    xs_02 = xs[0, :] - xs[2, :]
    xs_03 = xs[0, :] - xs[3, :]
    xs_12 = xs[1, :] - xs[2, :]
    xs_13 = xs[1, :] - xs[3, :]
    xs_23 = xs[2, :] - xs[3, :]

    # 预计算 Xi 与 xs 的差值
    X_i0 = Xi - xs[0, :]
    X_i1 = Xi - xs[1, :]
    X_i2 = Xi - xs[2, :]
    X_i3 = Xi - xs[3, :]

    # 计算 Lx
    Lx[0, :] = X_i1 * X_i2 * X_i3 / (xs_01 * xs_02 * xs_03)
    Lx[1, :] = -X_i0 * X_i2 * X_i3 / (xs_01 * xs_12 * xs_13)  # 注意此处为负数
    Lx[2, :] = X_i0 * X_i1 * X_i3 / (xs_02 * xs_12 * xs_23)
    Lx[3, :] = -X_i0 * X_i1 * X_i2 / (xs_03 * xs_13 * xs_23)  # 注意此处为负数

    # 预计算 ys 的差值
    ys_01 = ys[0, :] - ys[1, :]
    ys_02 = ys[0, :] - ys[2, :]
    ys_03 = ys[0, :] - ys[3, :]
    ys_12 = ys[1, :] - ys[2, :]
    ys_13 = ys[1, :] - ys[3, :]
    ys_23 = ys[2, :] - ys[3, :]

    # 预计算 Yi 与 ys 的差值
    Y_i0 = Yi - ys[0, :]
    Y_i1 = Yi - ys[1, :]
    Y_i2 = Yi - ys[2, :]
    Y_i3 = Yi - ys[3, :]

    # 计算 Ly
    Ly[0, :] = Y_i1 * Y_i2 * Y_i3 / (ys_01 * ys_02 * ys_03)
    Ly[1, :] = -Y_i0 * Y_i2 * Y_i3 / (ys_01 * ys_12 * ys_13)  # 注意此处为负数
    Ly[2, :] = Y_i0 * Y_i1 * Y_i3 / (ys_02 * ys_12 * ys_23)
    Ly[3, :] = -Y_i0 * Y_i1 * Y_i2 / (ys_03 * ys_13 * ys_23)  # 注意此处为负数

    # 计算 Lxy
    for i in range(4):
        for j in range(4):
            Lxy[i, j, :] = Lx[i, :] * Ly[j, :]

    return Lxy * zs


def Lagrange_interp_2D_vect_old(X, Y, Z, Xi, Yi):
    idx_array_x, OutRangeFlag = find_four_points_non_uniform_vector(X, Xi)
    idx_array_y, _ = find_four_points_non_uniform_vector(Y, Yi)

    xs = X[idx_array_x]
    ys = Y[idx_array_y]
    zs = Z[idx_array_x[:, None, :], idx_array_y[None, :, :]]

    interp_coefs = Interp_DoSum(xs, ys, zs, Xi, Yi)
    interp_value = np.sum(interp_coefs, axis=(0, 1))

    return interp_value, OutRangeFlag


@njit
def Lagrange_interp_2D_vect(X, Y, Z, Xi, Yi):
    idx_array_x, OutRangeFlag = find_four_points_non_uniform_vector(X, Xi)
    idx_array_y, _ = find_four_points_non_uniform_vector(Y, Yi)
    # idx_array_x, OutRangeFlag = find_four_points_non_uniform_vector_new(X, Xi)
    # idx_array_y, _ = find_four_points_non_uniform_vector_new(Y, Yi)

    xs = np.empty((idx_array_x.shape[0], idx_array_x.shape[1]), dtype=X.dtype)
    ys = np.empty((idx_array_y.shape[0], idx_array_y.shape[1]), dtype=Y.dtype)
    zs = np.empty((idx_array_x.shape[0], idx_array_y.shape[0], idx_array_y.shape[1]), dtype=Z.dtype)

    for i in range(idx_array_x.shape[0]):
        for j in range(idx_array_y.shape[0]):
            for k in range(idx_array_y.shape[1]):
                xs[i, k] = X[idx_array_x[i, k]]
                ys[j, k] = Y[idx_array_y[j, k]]
                zs[i, j, k] = Z[idx_array_x[i, k], idx_array_y[j, k]]  # 修改这里的索引逻辑

    interp_coefs = Interp_DoSum(xs, ys, zs, Xi, Yi)
    sum_over_axis_0 = np.sum(interp_coefs, axis=0)
    interp_value = np.sum(sum_over_axis_0, axis=0)

    return interp_value, OutRangeFlag


# @njit
def rk4_step(func, t, r, h, GlobalParameters, LocalParameters):
    # shape of r = number of particles * number of coordinates
    k1 = h * func(t, r, GlobalParameters, LocalParameters)
    k2 = h * func(t + 0.5 * h, r + 0.5 * k1, GlobalParameters, LocalParameters)
    k3 = h * func(t + 0.5 * h, r + 0.5 * k2, GlobalParameters, LocalParameters)
    k4 = h * func(t + h, r + k3, GlobalParameters, LocalParameters)
    # shape of k1,...,k4 = n*7
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


# segment cross checking
@njit
def direction_np_njit(a, b, c):
    ab = b - a
    ac = c - a
    return cross2d(ab, ac)


@njit
def on_segment_np_njit(a, b, c):
    return np.all(np.logical_and(np.minimum(a, b) <= c, c <= np.maximum(a, b)))


@njit
def segments_intersect_np_njit(arr1, arr2):
    a = arr1[:2]
    b = arr1[2:]
    c = arr2[:2]
    d = arr2[2:]

    LengthCD = np.sqrt((d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2)
    LengthAB = np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    d1 = direction_np_njit(c, d, a)
    d2 = direction_np_njit(c, d, b)
    d3 = direction_np_njit(a, b, c)
    d4 = direction_np_njit(a, b, d)

    L_A2CD = d1 / LengthCD
    L_B2CD = d2 / LengthCD
    L_C2AB = d3 / LengthAB
    L_D2AB = d4 / LengthAB

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB

    if d1 == 0 and on_segment_np_njit(c, d, a):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB
    if d2 == 0 and on_segment_np_njit(c, d, b):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB
    if d3 == 0 and on_segment_np_njit(a, b, c):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB
    if d4 == 0 and on_segment_np_njit(a, b, d):
        return True, L_A2CD, L_B2CD, L_C2AB, L_D2AB

    return False, L_A2CD, L_B2CD, L_C2AB, L_D2AB


@njit
def check_intersections_njit(segment_group_a, segment_group_b):
    na = len(segment_group_a)
    nb = len(segment_group_b)
    intersection_matrix = np.zeros((na, nb), dtype=types.boolean)
    Dist_matrix_A2CD = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_B2CD = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_C2AB = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_D2AB = np.zeros((na, nb), dtype=types.float64)
    for i, seg_a in enumerate(segment_group_a):
        for j, seg_b in enumerate(segment_group_b):
            (intersection_matrix[i, j], Dist_matrix_A2CD[i, j],
             Dist_matrix_B2CD[i, j], Dist_matrix_C2AB[i, j],
             Dist_matrix_D2AB[i, j]) = segments_intersect_np_njit(seg_a, seg_b)
    return intersection_matrix, Dist_matrix_A2CD, Dist_matrix_B2CD, Dist_matrix_C2AB, Dist_matrix_D2AB

@njit
def CheckIntersect_njit_ParticleCoord(r_PreStep, r_PostStep, segment_group_b):
    r_pre, fi_pre = r_PreStep[:, 0], r_PreStep[:, 4]
    r_post, fi_post = r_PostStep[:, 0], r_PostStep[:, 4]

    x_pre, y_pre = r_pre * np.cos(fi_pre), r_pre * np.sin(fi_pre)
    x_post, y_post = r_post * np.cos(fi_post), r_post * np.sin(fi_post)

    segment_group_a = np.column_stack((x_pre, y_pre, x_post, y_post))

    na = len(segment_group_a)
    nb = len(segment_group_b)
    nn = r_PreStep.shape[1]
    intersection_matrix = np.zeros((na, nb), dtype=types.boolean)
    Dist_matrix_A2CD = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_B2CD = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_C2AB = np.zeros((na, nb), dtype=types.float64)
    Dist_matrix_D2AB = np.zeros((na, nb), dtype=types.float64)

    r_CentralStep = np.zeros((na, nb, nn))

    for i, seg_a in enumerate(segment_group_a):
        for j, seg_b in enumerate(segment_group_b):
            (intersection_matrix[i, j], Dist_matrix_A2CD[i, j],
             Dist_matrix_B2CD[i, j], Dist_matrix_C2AB[i, j],
             Dist_matrix_D2AB[i, j]) = segments_intersect_np_njit(seg_a, seg_b)
            if intersection_matrix[i, j]:
                AoOverBo = (np.abs(Dist_matrix_A2CD[i, j]) / (
                        np.abs(Dist_matrix_A2CD[i, j]) + np.abs(Dist_matrix_B2CD[i, j])))
                r_CentralStep[i, j, :] = r_PreStep[i, :] + AoOverBo * (r_PostStep[i, :] - r_PreStep[i, :])

    return intersection_matrix, r_CentralStep, Dist_matrix_A2CD, Dist_matrix_B2CD, Dist_matrix_C2AB, Dist_matrix_D2AB

@njit
def stack_to_matrix(stack_matrix, stack_ids, input_matrix, input_ids):
    # stack_matrix: 已有的堆叠矩阵 (n, 10, t)
    # input_matrix: 要堆叠的粒子坐标 (n, 10)
    # input_ids: 要堆叠的粒子id号 (m, )
    # stack_ids: 每个粒子的堆叠次数 (n, )

    n = stack_matrix.shape[0]  # 粒子总数
    for particle_id in input_ids:
        # 确定当前粒子的堆叠层数
        layer_id = stack_ids[particle_id]  # 使用 stack_ids 跟踪堆叠次数

        # 如果层数超出当前矩阵的层数，扩展矩阵
        if layer_id >= stack_matrix.shape[2]:
            # 扩展一个层次
            stack_matrix = np.append(stack_matrix, -np.ones((n, 10, stack_matrix.shape[2])), axis=2)

        # 堆叠粒子
        stack_matrix[particle_id, :, layer_id] = input_matrix[particle_id, :]

        # 更新该粒子的堆叠次数
        stack_ids[particle_id] += 1

    return stack_matrix, stack_ids

@njit
def v2Ek_fast(v):
    c = 2.99792458e8
    q = 1.60217662e-19
    E0_J = 938.2723e6 * q
    beta = v / c
    gamma = 1 / np.sqrt(1 - beta ** 2)
    Ek_J = (gamma - 1) * E0_J
    Etotal_J = E0_J + Ek_J
    Ek_MeV = Ek_J / q / 1e6
    return Ek_MeV, Etotal_J

@njit
def v2Ek_J_fast(v):
    c = 2.99792458e8
    q = 1.60217662e-19
    E0_J = 938.2723e6 * q
    Etotal_J = E0_J * c / np.sqrt(c**2 - v ** 2)
    return Etotal_J


@njit
def stack_to_matrix(stack_matrix, stack_ids, input_matrix, input_ids):
    # stack_matrix: 已有的堆叠矩阵 (n, 10, t)
    # input_matrix: 要堆叠的粒子坐标 (n, 10)
    # input_ids: 要堆叠的粒子id号 (m, )
    # stack_ids: 每个粒子的堆叠次数 (n, )

    n = stack_matrix.shape[0]  # 粒子总数
    m = stack_matrix.shape[1]  # dims
    for particle_id in input_ids:
        # 确定当前粒子的堆叠层数
        layer_id = stack_ids[particle_id]  # 使用 stack_ids 跟踪堆叠次数

        # 如果层数超出当前矩阵的层数，扩展矩阵
        if layer_id >= stack_matrix.shape[2]:
            # 扩展一个层次
            stack_matrix = np.append(stack_matrix, -np.ones((n, m, stack_matrix.shape[2])), axis=2)

        # 堆叠粒子
        stack_matrix[particle_id, :, layer_id] = input_matrix[particle_id, :]

        # 更新该粒子的堆叠次数
        stack_ids[particle_id] += 1

    return stack_matrix, stack_ids


# @njit
def linear_interpolation_njit(X, Y, xi):

    idx_array, _ = find_two_points_non_uniform_vect_njit(X, xi)
    x = X[idx_array]
    y = Y[idx_array]
    x0, x1, y0, y1 = x[0], x[1], y[0], y[1]
    return y0 + (xi - x0) / (x1 - x0) * (y1 - y0)


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

        x0_z0_interp = (yi_current - y_left) / (y_right - y_left) * x0_10 + (y_right - yi_current) / (y_right - y_left) * x0_00
        x0_z1_interp = (yi_current - y_left) / (y_right - y_left) * x0_11 + (y_right - yi_current) / (y_right - y_left) * x0_01
        x0_interp = (zi_current - z_left) / (z_right - z_left) * x0_z1_interp + (z_right - zi_current) / (z_right - z_left) * x0_z0_interp

        x1_z0_interp = (yi_current - y_left) / (y_right - y_left) * x1_10 + (y_right - yi_current) / (y_right - y_left) * x1_00
        x1_z1_interp = (yi_current - y_left) / (y_right - y_left) * x1_11 + (y_right - yi_current) / (y_right - y_left) * x1_01
        x1_interp = (zi_current - z_left) / (z_right - z_left) * x1_z1_interp + (z_right - zi_current) / (z_right - z_left) * x1_z0_interp

        interp_values[idxs] = (xi_current - x_left) / (x_right - x_left) * x1_interp + (x_right - xi_current) / (x_right - x_left) * x0_interp

    return interp_values
