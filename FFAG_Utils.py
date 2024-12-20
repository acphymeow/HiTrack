import numpy as np
import os
import shutil
import copy


def StepInterpNew(pre, post, fi_threshold):
    # Interp on the adjacent time steps

    rownum, colnum = np.size(pre, 0), np.size(pre, 1)

    x0 = pre[:, -1].reshape((-1, 1))
    x1 = post[:, -1].reshape((-1, 1))
    x = fi_threshold.reshape((-1, 1))
    y0 = copy.deepcopy(pre)
    y1 = copy.deepcopy(post)

    x0 = np.tile(x0, (1, colnum))
    x1 = np.tile(x1, (1, colnum))
    x = np.tile(x, (1, colnum))

    y = (x - x0) / (x1 - x0) * y1 + (x - x1) / (x0 - x1) * y0
    return y


class FFAG_GeometryCalc:
    def __init__(self):
        pass

    def doBoundingBoxesOverlap(self, A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y):
        # 快速排斥
        # A ---> (A1x, A1y, A2x, A2y)
        # B ---> (B1x, B1y, B2x, B2y)
        NonOverlap = ((np.minimum(A1x, A2x) >= np.maximum(B1x, B2x)) +
                      (np.maximum(A1x, A2x) <= np.minimum(B1x, B2x)) +
                      (np.minimum(A1y, A2y) >= np.maximum(B1y, B2y)) +
                      (np.maximum(A1y, A2y) <= np.minimum(B1y, B2y)))

        return NonOverlap

    # def doBoundingBoxesOverlap_Fast(self, A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y):
    #     # 快速排斥
    #     # A ---> (A1x, A1y, A2x, A2y)
    #     # B ---> (B1x, B1y, B2x, B2y)
    #     # 4个边界都不重叠，返回NonOverlap为True
    #     NonOverlap_xleft = (np.minimum(A1x, A2x) >= np.maximum(B1x, B2x))
    #     NonOverlap_xright = (np.maximum(A1x, A2x) <= np.minimum(B1x, B2x))
    #     NonOverlap_yup = (np.minimum(A1y, A2y) >= np.maximum(B1y, B2y))
    #     NonOverlap_ydown = (np.maximum(A1y, A2y) <= np.minimum(B1y, B2y))
    #     NonOverlap = NonOverlap_xleft + NonOverlap_xright + NonOverlap_yup + NonOverlap_ydown
    #     return NonOverlap

    def doLinesIntersect(self, A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y):
        # 跨立实验

        # The length of A, B
        LengthA = np.sqrt((A2x - A1x) ** 2 + (A2y - A1y) ** 2)
        LengthB = np.sqrt((B2x - B1x) ** 2 + (B2y - B1y) ** 2)
        # (B1B2 cross B1A1)/ norm(B1B2) = distance from A1 to B1B2
        # (B1B2 cross B1A2)/ norm(B1B2) = distance from A2 to B1B2

        # 以B为基准，判断点A1A2是否在线段B两侧
        # B1B2 cross B1A1
        # B1B2=((B2[0]-B1[0]), (B2[1]-B1[1]))
        # B1A1=((A1[0]-B1[0]), (A1[1]-B1[1]))
        cross_product_A1 = (B2x - B1x) * (A1y - B1y) - (B2y - B1y) * (A1x - B1x)
        # D_A1_to_B = cross_product_A1 / LengthB
        D_A1_to_B = np.divide(cross_product_A1, LengthB,
                              out=np.full_like(LengthB, np.nan, dtype=float), where=LengthB != 0)

        # B1B2 cross B1A2
        # B1B2=((B2[0]-B1[0]), (B2[1]-B1[1]))
        # B1A2=((A2[0]-B1[0]), (A2[1]-B1[1]))
        cross_product_A2 = (B2x - B1x) * (A2y - B1y) - (B2y - B1y) * (A2x - B1x)
        # D_A2_to_B = cross_product_A2 / LengthB
        D_A2_to_B = np.divide(cross_product_A2, LengthB,
                              out=np.full_like(LengthB, np.nan, dtype=float), where=LengthB != 0)

        # 以A为基准，判断点B1B2是否在线段A两侧
        # A1A2 cross A1B1
        # A1A2 = ((A2[0]-A1[0]), (A2[1]-A1[1]))
        # A1B1 = ((B1[0]-A1[0]), (B1[1]-A1[1]))
        cross_product_B1 = (A2x - A1x) * (B1y - A1y) - (A2y - A1y) * (B1x - A1x)
        # D_B1_to_A = cross_product_B1 / LengthA
        D_B1_to_A = np.divide(cross_product_B1, LengthA,
                              out=np.full_like(LengthA, np.nan, dtype=float), where=LengthA != 0)

        # A1A2 cross A1B2
        # A1A2 = ((A2[0] - A1[0]), (A2[1] - A1[1]))
        # A1B2 = ((B2[0] - A1[0]), (B2[1] - A1[1]))
        cross_product_B2 = (A2x - A1x) * (B2y - A1y) - (A2y - A1y) * (B2x - A1x)
        # D_B2_to_A = cross_product_B2 / LengthA
        D_B2_to_A = np.divide(cross_product_B2, LengthA,
                              out=np.full_like(LengthA, np.nan, dtype=float), where=LengthA != 0)

        crossA = cross_product_A1 * cross_product_A2
        crossB = cross_product_B1 * cross_product_B2

        # if crossA<0 and crossB<0, return true
        # else return false
        flagA = crossA < 0
        flagB = crossB < 0
        flag = flagA * flagB

        return flag, D_A1_to_B, D_A2_to_B, D_B1_to_A, D_B2_to_A

    def FindIntersectionVect(self, A, B):
        """
        同时判断多个线段与多个线段是否相交,A,B代表多条线段
        # A ---> (A1x, A1y, A2x, A2y)
        # B ---> (B1x, B1y, B2x, B2y)
        # test code: test_cross_spiral.py
        """
        # reshape A, B
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)

        RowsOfA = np.size(A, 0)
        RowsOfB = np.size(B, 0)

        A = np.repeat(A, RowsOfB, axis=0)
        B = np.tile(B, (RowsOfA, 1))

        (A1x, A1y, A2x, A2y) = (A[:, 0], A[:, 1], A[:, 2], A[:, 3])
        (B1x, B1y, B2x, B2y) = (B[:, 0], B[:, 1], B[:, 2], B[:, 3])

        OverlapFlag = ~self.doBoundingBoxesOverlap(A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y)
        # 并不需要每个step进行doLinesIntersect, 若OverlapFlag全为False则跳过该步骤
        NoneParticleOverlap = np.sum(OverlapFlag) == 0

        if NoneParticleOverlap:
            IntersectFlag, D_A1_B, D_A2_B, D_B1_A, D_B2_A = (
                np.full_like(OverlapFlag, False, dtype=bool),
                np.full_like(OverlapFlag, False, dtype=bool),
                np.full_like(OverlapFlag, False, dtype=bool),
                np.full_like(OverlapFlag, False, dtype=bool),
                np.full_like(OverlapFlag, False, dtype=bool))
        else:
            IntersectFlag, D_A1_B, D_A2_B, D_B1_A, D_B2_A = (
                self.doLinesIntersect(A1x, A1y, A2x, A2y, B1x, B1y, B2x, B2y))


        Flag = OverlapFlag * IntersectFlag
        Flag_reshape = np.reshape(Flag, (RowsOfA, RowsOfB))
        D_A1_B_matrix = np.reshape(D_A1_B, (RowsOfA, RowsOfB))
        D_A2_B_matrix = np.reshape(D_A2_B, (RowsOfA, RowsOfB))
        D_B1_A_matrix = np.reshape(D_B1_A, (RowsOfA, RowsOfB))
        D_B2_A_matrix = np.reshape(D_B2_A, (RowsOfA, RowsOfB))

        return Flag_reshape, D_A1_B_matrix, D_A2_B_matrix, D_B1_A_matrix, D_B2_A_matrix


    def GetVectAngle(self, VectA, VectB):
        # get angles between VectA and VectB.
        # dimensions of inputs: (n,2) ndarray
        dotAB = VectA[:, 0] * VectB[:, 0] + VectA[:, 1] * VectB[:, 1]
        normAB = np.sqrt(VectA[:, 0] ** 2 + VectA[:, 1] ** 2) * np.sqrt(VectB[:, 0] ** 2 + VectB[:, 1] ** 2)
        # angle_radians = np.arccos(dotAB / normAB)
        angle_radians = np.arccos(np.clip((dotAB / normAB), -1, 1))
        angle_degree = np.rad2deg(angle_radians)
        CrossValue = VectA[:, 0] * VectB[:, 1] - VectA[:, 1] * VectB[:, 0]
        Sign = np.ones_like(CrossValue)
        Sign[CrossValue < 0] = -1
        angle_radians_WithSign, angle_degree_WithSign = angle_radians * Sign, angle_degree * Sign
        return angle_radians_WithSign, angle_degree_WithSign

    def CheckCrossFiVectNew(self, fi, fi_threshold):
        """
        检查当前step是否穿越了给定方位角
        test code: test_cross_Fi2.py
        """
        azimuth_PreStep, azimuth_CurrentStep = fi[:, -2], fi[:, -1]
        azimuth_threshold = np.ones_like(azimuth_PreStep) * fi_threshold

        # 映射到单位圆上3点
        a = np.column_stack((np.cos(azimuth_PreStep), np.sin(azimuth_PreStep)))
        b = np.column_stack((np.cos(azimuth_CurrentStep), np.sin(azimuth_CurrentStep)))
        c = np.column_stack((np.cos(azimuth_threshold), np.sin(azimuth_threshold)))

        # 判断点c是否在圆弧上
        angle_ab, _ = self.GetVectAngle(a, b)
        angle_ac, _ = self.GetVectAngle(a, c)
        is_on_arc = (angle_ac - angle_ab) * (angle_ac - 0) < 0

        angle_a = np.zeros_like(angle_ab)
        angle_b, angle_c = angle_a + angle_ab, angle_a + angle_ac

        return is_on_arc, angle_a, angle_b, angle_c

    def calculate_angle(self, point1, point2, point3):
        vector1 = point1 - point2
        vector2 = point3 - point2

        # dot_product = np.dot(vector1, vector2)
        # norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        dot_product_vect = np.sum(vector1 * vector2, axis=1)
        norm_product_vect = np.sqrt(vector1[:, 0] ** 2 + vector1[:, 1] ** 2) * np.sqrt(
            vector2[:, 0] ** 2 + vector2[:, 1] ** 2)

        angle = np.arccos(dot_product_vect / norm_product_vect)
        angleWithSign = copy.deepcopy(angle)

        # a1, a2, a3 = vector1[0], vector1[1], 0.0
        # b1, b2, b3 = vector2[0], vector2[1], 0.0
        vector_cross = vector1[:, 0] * vector2[:, 1] - vector1[:, 1] * vector2[:, 0]

        # flag1 = vector_cross != 0
        flag1 = ~np.isclose(vector_cross, 0)

        flag2_1 = angle > np.deg2rad(179)
        flag3_1 = angle < np.deg2rad(1)

        angleWithSign[flag1] = angle[flag1] * vector_cross[flag1] / np.abs(vector_cross[flag1])

        flag2 = (~flag1) & flag2_1
        flag3 = (~flag1) & flag3_1

        angleWithSign[flag2] = np.pi
        angleWithSign[flag3] = 0

        return angleWithSign

    def point_in_convex_polygon(self, polygon_points, point_to_check):
        """
        判断点是否在凸多边形内部

        参数：
        scatter_points：散点的坐标，二维NumPy数组，每一行代表一个散点的坐标。
        point_to_check：待检测点的坐标，一维NumPy数组或列表。

        返回值：
        如果待检测点在凸多边形内部，返回True；否则，返回False。
        """
        # 将散点坐标转换为NumPy数组
        polygon_points_closed = copy.deepcopy(polygon_points)
        polygon_points_closed = np.vstack((polygon_points_closed, polygon_points_closed[0, :]))

        NumOfPointsToCheck = np.size(point_to_check, 0)
        NumOfPointsToPolygon = np.size(polygon_points, 0)

        polygon_points_left = polygon_points_closed[:-1, :]
        polygon_points_right = polygon_points_closed[1:, :]

        point_i_repeat = np.repeat(point_to_check, NumOfPointsToPolygon, axis=0)
        polygon_points_left_repeat = np.tile(polygon_points_left, (NumOfPointsToCheck, 1))
        polygon_points_right_repeat = np.tile(polygon_points_right, (NumOfPointsToCheck, 1))

        angleWithSign = self.calculate_angle(polygon_points_left_repeat, point_i_repeat, polygon_points_right_repeat)
        angleWithSign_splits = np.split(angleWithSign, NumOfPointsToCheck)
        sums = np.sum(angleWithSign_splits, axis=1)

        flag_in_polygon = np.abs(sums) > np.deg2rad(350.0)
        return flag_in_polygon


class FFAG_SegmentTools:
    def __init__(self):
        pass

    def MatrixDotCross(self, matrix1, matrix2):
        DotValue = matrix1[:, 0] * matrix2[:, 0] + matrix1[:, 1] * matrix2[:, 1]
        CrossValue = matrix1[:, 0] * matrix2[:, 1] - matrix1[:, 1] * matrix2[:, 0]
        return DotValue, CrossValue

    def MatrixNorm(self, matrix1):
        norm1 = np.sqrt(matrix1[:, 0] ** 2 + matrix1[:, 1] ** 2)
        return norm1

    def CentralLine2Segments(self, cl):
        # 将曲线上的连续散点转换为折线端点（n,2）--->(n-1,4)
        cl_mid = cl[1:-1, :]
        cl_mid_repeat = np.repeat(cl_mid, 2, axis=0)
        cl_mid_repeat_flat = np.reshape(cl_mid_repeat, (1, -1))
        cl_mid_repeat_flat_full = (
            np.column_stack((np.atleast_2d(cl[0]), cl_mid_repeat_flat, np.atleast_2d(cl[-1]))))
        cl_segment = np.reshape(cl_mid_repeat_flat_full, (-1, 4))
        return cl_segment

    def SegmentPointMin(self, line1, sample_point):
        # find the minimum distance from a point to a segment
        # test code : test_segment_min.py
        cl_segment = self.CentralLine2Segments(line1)

        # 计算直线上的向量
        point1, point2 = cl_segment[:, 0:2], cl_segment[:, 2:4]  # 折线的起始点、终点
        line_vector = point2 - point1
        line_vector = np.atleast_2d(line_vector)

        # 计算样本点到直线的向量
        sample_vector = sample_point - point1
        sample_vector = np.atleast_2d(sample_vector)

        # 计算最短距离的点
        ratio = self.MatrixDotCross(sample_vector, line_vector)[0] / self.MatrixDotCross(line_vector, line_vector)[0]
        flag0 = ratio < 0
        flag1 = ratio > 1
        ratio[flag0], ratio[flag1] = 0, 1
        ratio = np.reshape(np.repeat(ratio, 2, axis=0), (-1, 2))
        projection_point = point1 + ratio * line_vector

        # 计算全局最短距离的点
        D_proj_samp = self.MatrixNorm(projection_point - sample_point)
        min_arg = np.argmin(D_proj_samp)
        projection_min, D_min = projection_point[min_arg, :], D_proj_samp[min_arg]

        return projection_min, D_min, projection_point

    def SegmentGridMin(self, line1, grid_points_x, grid_points_y):
        # find the minimum distance from grid points to a segment
        # test code : test_segment_min.py
        cl_segment = self.CentralLine2Segments(line1)
        cl_point_s, cl_point_e = cl_segment[:, 0:2], cl_segment[:, 2:4]  # 折线的起始点、终点

        n_grid_x, n_grid_y = np.size(grid_points_x, 0), np.size(grid_points_x, 1)
        n_line_d = np.size(cl_segment, 0)

        # 分离折线的xy坐标
        cl_point_s_x_raw, cl_point_s_y_raw = cl_point_s[:, 0], cl_point_s[:, 1]
        cl_point_e_x_raw, cl_point_e_y_raw = cl_point_e[:, 0], cl_point_e[:, 1]
        # 切向向量
        tan_direct_x = cl_point_e_x_raw - cl_point_s_x_raw
        tan_direct_y = cl_point_e_y_raw - cl_point_s_y_raw
        # broadcast
        cl_point_s_x = np.tile(cl_point_s_x_raw[np.newaxis, np.newaxis, :], (n_grid_x, n_grid_y, 1))
        cl_point_s_y = np.tile(cl_point_s_y_raw[np.newaxis, np.newaxis, :], (n_grid_x, n_grid_y, 1))
        cl_point_e_x = np.tile(cl_point_e_x_raw[np.newaxis, np.newaxis, :], (n_grid_x, n_grid_y, 1))
        cl_point_e_y = np.tile(cl_point_e_y_raw[np.newaxis, np.newaxis, :], (n_grid_x, n_grid_y, 1))

        grid_x_broadcast = np.tile(grid_points_x[..., np.newaxis], (1, 1, n_line_d))
        grid_y_broadcast = np.tile(grid_points_y[..., np.newaxis], (1, 1, n_line_d))

        # 计算网格点到折线的向量(折线的每段有2点，2个二维坐标，4个xy)
        line_vector_x = cl_point_e_x - cl_point_s_x
        line_vector_y = cl_point_e_y - cl_point_s_y
        grid_vector_s_x = grid_x_broadcast - cl_point_s_x
        grid_vector_s_y = grid_y_broadcast - cl_point_s_y

        # calculate dot(line_vector, grid_vector)
        Dot_line_line = line_vector_x * line_vector_x + line_vector_y * line_vector_y
        Dot_grid_line = grid_vector_s_x * line_vector_x + grid_vector_s_y * line_vector_y

        # 计算到折线各分段最短距离
        ratio = Dot_grid_line / Dot_line_line
        flag0 = ratio < 0
        flag1 = ratio > 1
        ratio[flag0], ratio[flag1] = 0, 1
        projection_point_x = cl_point_s_x + ratio * line_vector_x
        projection_point_y = cl_point_s_y + ratio * line_vector_y

        # 计算全局最短距离的点
        D_proj_samp = np.sqrt(
            (projection_point_x - grid_x_broadcast) ** 2 + (projection_point_y - grid_y_broadcast) ** 2)
        min_arg_axis2 = np.argmin(D_proj_samp, axis=2)
        min_arg_axis0, min_arg_axis1 = np.indices((n_grid_x, n_grid_y))[0], np.indices((n_grid_x, n_grid_y))[1]

        projection_x_min = projection_point_x[min_arg_axis0, min_arg_axis1, min_arg_axis2]
        projection_y_min = projection_point_y[min_arg_axis0, min_arg_axis1, min_arg_axis2]
        D_min = D_proj_samp[min_arg_axis0, min_arg_axis1, min_arg_axis2]

        # find the tang direction of the projection points
        projection_tan_x = tan_direct_x[min_arg_axis2]
        projection_tan_y = tan_direct_y[min_arg_axis2]

        return projection_x_min, projection_y_min, D_min, projection_point_x, projection_point_y, projection_tan_x, projection_tan_y

    def find_circle_line_intersections(self, line_points, circle_x, circle_y, radius):
        # 二维平面上有若干散点构成的折线段，给定一个半径值R，求以R为半径的圆与折线的交点坐标
        # test code: test_segment_circle_cross.py
        intersections = []
        for i in range(len(line_points) - 1):
            x1, y1 = line_points[i]
            x2, y2 = line_points[i + 1]
            # 计算线段的方向向量
            dx = x2 - x1
            dy = y2 - y1

            # 计算直线的参数
            A = dx ** 2 + dy ** 2
            B = 2 * (dx * (x1 - circle_x) + dy * (y1 - circle_y))
            C = (x1 - circle_x) ** 2 + (y1 - circle_y) ** 2 - radius ** 2

            # 计算判别式
            discriminant = B ** 2 - 4 * A * C

            # 如果判别式小于零，没有交点
            if discriminant < 0:
                continue

            # 计算两个交点
            t1 = (-B + np.sqrt(discriminant)) / (2 * A)
            t2 = (-B - np.sqrt(discriminant)) / (2 * A)

            # 检查是否交点在线段内
            if 0 <= t1 <= 1:
                intersection_x1 = x1 + t1 * dx
                intersection_y1 = y1 + t1 * dy
                intersections.append((intersection_x1, intersection_y1))

            if 0 <= t2 <= 1:
                intersection_x2 = x1 + t2 * dx
                intersection_y2 = y1 + t2 * dy
                intersections.append((intersection_x2, intersection_y2))

        return intersections

    def segment_interp(self, points, ratio):
        """
        计算折线的长度并进行比例剖分，返回插值点和输入点的坐标，并将它们按距离进行排序。
        test code: test_segment_interp.py

        Parameters:
        points (numpy.ndarray): 二维数组，包含按序排序的散点坐标，每行表示一个点。
        ratio (numpy.ndarray): 包含要进行等距剖分的位置的比例值。

        Returns:
        xyd_interp (numpy.ndarray): 包含等距剖分的插值点坐标和对应的距离。
        xyd_input (numpy.ndarray): 包含输入的散点坐标和对应的距离。
        xyd_all_sorted (numpy.ndarray): 包含所有点（插值点和输入点）按距离排序后的坐标。

        """
        flag_ratio_less_0 = ratio <= 0
        flag_ratio_larger_1 = ratio >= 1
        ratio[flag_ratio_less_0] = 0.000001
        ratio[flag_ratio_larger_1] = 1

        # 计算每个线段的长度
        length = np.linalg.norm(np.diff(points, axis=0), axis=1)

        # 计算每个点到起点的距离
        distances = np.cumsum(np.insert(length, 0, 0))

        # 计算整条线的总长度
        total_length = np.sum(length)

        # 根据所给的比例值计算插值点
        idx_e = np.searchsorted(distances, total_length * ratio)
        idx_s = idx_e - 1

        distances_interp = total_length * ratio
        mod_length_ratio = (distances_interp - distances[idx_s]) / length[idx_s]
        mod_length_ratio_repeat = np.repeat(mod_length_ratio, 2, axis=0)
        mod_length_ratio_reshape = np.reshape(mod_length_ratio_repeat, (-1, 2))
        xy_s, xy_e = points[idx_s, :], points[idx_e, :]

        xy_interp = xy_s + (xy_e - xy_s) * mod_length_ratio_reshape

        xyd_interp = np.column_stack((xy_interp, distances_interp))
        xyd_input = np.column_stack((points, distances))
        xyd_all = np.row_stack((xyd_input, xyd_interp))

        # 获取根据最后一列（distance）排序后的索引
        sorted_indices = np.argsort(xyd_all[:, -1])
        # 使用索引对 xyd_all 进行排序
        xyd_all_sorted = xyd_all[sorted_indices]

        return xyd_interp, xyd_input, xyd_all_sorted


def dynamic_insert_3D(matrix_a, indices_i, indices_j, value_b):
    # indices_i and indices_j are (1, n) arrays
    # value_b is a (n, l) array
    # matrix_a is a (j, k, l) array
    rows, cols, deepth = np.size(matrix_a, 0), np.size(matrix_a, 1), np.size(matrix_a, 2)
    # steps, number of particles, dims
    max_i = np.max(indices_i)

    # Extend matrix_a if needed
    if max_i + 1 > rows:
        extension = np.ones((4, cols, deepth)) * (-1)
        matrix_a = np.concatenate((matrix_a, extension), axis=0)

    matrix_a[indices_i, indices_j, :] = value_b

    return matrix_a


def append_with_limit(lst, new_data, limit):
    lst.append(new_data)
    if len(lst) > limit:
        del lst[0]


def concatenate_with_limit(tr_points_AllSteps, tr_points_ThisStep, max_length):
    # 将tr_points_ThisStep附加到tr_points_AllSteps
    tr_points_AllSteps = np.concatenate((tr_points_AllSteps, tr_points_ThisStep[np.newaxis, :, :]), axis=0)

    # 如果tr_points_AllSteps的长度已经超过了max_length，删除前面的层
    CurrentLength = np.size(tr_points_AllSteps, 0)
    if CurrentLength > max_length:
        tr_points_AllSteps = tr_points_AllSteps[-max_length:, :, :]

    return tr_points_AllSteps


class FFAG_FileOperation:
    def __init__(self):
        pass

    def CreatAEmptyFold(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"The folder '{folder_path}' has been created.")
        else:
            print(f"The folder '{folder_path}' already exists.")
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete file: {file_path} ({e})")
            print(f"The folder '{folder_path}' has been cleared.")

    def CreatAEmptyFold_safe(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"The folder '{folder_path}' has been created.")
        else:
            user_input = input(f"Are you sure you want to clear the folder '{folder_path}'? (Y/N): ")

            if user_input.upper() == 'Y':
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        else:
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Failed to delete file: {file_path} ({e})")
                print(f"The folder '{folder_path}' has been cleared.")
            else:
                print("Operation canceled by the user.")

    def CreatAEmptyFoldUncovered(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"The folder '{folder_path}' has been created.")

        i = 1
        while True:
            new_folder_name = str(i)
            new_folder_path = os.path.normpath(os.path.join(folder_path, new_folder_name))
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                print(f"Created subfolder '{new_folder_name}' in '{folder_path}'.")
                break
            else:
                i += 1

        return new_folder_path


class FFAG_DynamicMatrix:
    """
    test code: test_dynamicMatrixWithLimit.py
    """
    def __init__(self, dim2, dim3):
        self.MatrixCapability = 16
        self.Matrix3D = -1 * np.ones((self.MatrixCapability, dim2, dim3))
        self.Insert_index = 0
        self.dim2, self.dim3 = dim2, dim3
        self.Matrix3DValid = None

    def find_power_to_exceed(self, a, b):
        # 计算MatrixCapability翻倍次数，向上取整
        power = int(np.ceil(np.log2(b / a)))
        bDoubleN = a * (2 ** power)
        bExpand = bDoubleN - a
        return power, bDoubleN, bExpand

    def dynamic_insert(self, Matrix2D):
        # Expand matrix if needed
        if self.Insert_index > self.MatrixCapability - 1:
            _, _, CapabilityExpand = self.find_power_to_exceed(self.MatrixCapability, self.Insert_index+1)
            MatrixExtend = -1 * np.ones((CapabilityExpand, self.dim2, self.dim3))
            self.Matrix3D = np.concatenate((self.Matrix3D, MatrixExtend), axis=0)
            self.MatrixCapability += CapabilityExpand

        self.Matrix3D[self.Insert_index, :, :] = Matrix2D
        self.Insert_index += 1

    def RestoreMatrix(self,):
        self.Matrix3DValid = self.Matrix3D[:self.Insert_index, :, :]
        return self.Matrix3DValid


class FFAG_DynamicMatrixWithLimit:

    def __init__(self, dim2, dim3, SizeLimit):
        self.SizeLimit = SizeLimit
        self.Matrix3D = -1 * np.ones((SizeLimit, dim2, dim3))
        self.CurrentTurns = 0
        self.CurrentRows = 0
        self.Insert_index = 0
        self.Matrix3DValid = None

    def dynamic_insert_withLimit(self, Matrix2D):
        self.CurrentTurns = int(self.Insert_index / self.SizeLimit)
        self.CurrentRows = self.Insert_index - self.CurrentTurns * self.SizeLimit
        self.Matrix3D[self.CurrentRows, :, :] = Matrix2D
        self.Insert_index += 1

    def RestoreMatrix(self):
        if self.CurrentTurns == 0:
            self.Matrix3DValid = self.Matrix3D[:self.CurrentRows + 1, :, :]
        else:
            self.PrePart = copy.deepcopy(self.Matrix3D[self.CurrentRows + 1:, :, :])
            self.PostPart = copy.deepcopy(self.Matrix3D[:self.CurrentRows + 1, :, :])
            self.Matrix3DValid = np.row_stack((self.PrePart, self.PostPart))
        return self.Matrix3DValid