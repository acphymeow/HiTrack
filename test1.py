import numpy as np
import matplotlib.pyplot as plt
from FFAG_Bunch import FFAG_Bunch
from FFAG_ParasAndConversion import FFAG_GlobalParameters
from FFAG_SC import Bunch_SC_Calculator


# 生成 C 形的环形长束团分布
def generate_c_shaped_ring_bunch(num_particles):
    """
    生成 C 形的环形长束团分布。

    参数：
    - num_particles: 粒子数

    返回：
    - particles: 粒子坐标矩阵 (num_particles x 15)，包含 r, vr, z, vz, fi, vfi
    """
    # fi 在 0 到 160 度之间
    fi = np.deg2rad(np.linspace(0, 160, num_particles))

    # r 在 0.95 到 1.05 m 之间随机生成
    r = np.random.uniform(0.95, 1.05, num_particles)

    # z 在 -3 到 3 cm 之间随机生成
    z = np.random.uniform(-0.03, 0.03, num_particles)  # z 方向长度 (m)

    # 随机生成速度分量
    vr = np.random.uniform(-0.01, 0.01, num_particles)  # 径向速度 (m/s)
    vz = np.random.uniform(-0.01, 0.01, num_particles)  # z 方向速度 (m/s)
    vfi = np.random.uniform(-0.01, 0.01, num_particles)  # 方位角速度 (rad/s)

    # 初始化其他属性
    Ek = np.full(num_particles, 1.0)  # 初始动能
    inj_t = np.zeros(num_particles)  # 注入时间
    survive_flag = np.ones(num_particles)
    RF_phase = np.zeros(num_particles)
    Esc_r = np.zeros(num_particles)
    Esc_z = np.zeros(num_particles)
    Esc_fi = np.zeros(num_particles)
    Bunch_ID = np.zeros(num_particles)
    Local_ID = np.arange(num_particles)
    Global_ID = np.arange(num_particles)

    # 组合粒子分布矩阵
    particles = np.column_stack((
        r, vr, z, vz, fi, vfi, inj_t, survive_flag, RF_phase,
        Esc_r, Esc_z, Esc_fi, Bunch_ID, Local_ID, Global_ID
    ))

    return particles


# 主函数
def main():
    num_particles = 1000000  # 粒子数

    # 生成粒子分布
    particles = generate_c_shaped_ring_bunch(num_particles)

    # 创建 FFAG_Bunch 对象
    # bunch = FFAG_Bunch(particles)
    BunchObj = FFAG_Bunch(particles, marcosize=10000)

    BunchObj.UpdateAndInjectParticles(0.0)
    BunchObj.UpdateGlobalBunch()
    BunchObj.Get_SC_Grid_para_global()
    BunchObj.DistributeChargeGlobal(60, 64, 68)


    # 计算空间电荷电场
    Ex_distribution, Ey_distribution, Ez_distribution, Er_interp, Efi_interp, Ez_interp, voltage_distribution_out, charge_distribution_out = Bunch_SC_Calculator(BunchObj)

    # 可视化粒子分布和电场分量
    visualize_bunch_and_fields(BunchObj, Ex_distribution, Ey_distribution, voltage_distribution_out)


def visualize_bunch_and_fields(bunch, Ex_distribution, Ey_distribution, charge_distribution):
    """
    可视化粒子分布和电场分量。
    """
    r = bunch.LocalBunch[:, bunch.BunchAttribute.Attribute['r']]
    fi = bunch.LocalBunch[:, bunch.BunchAttribute.Attribute['fi']]
    z = bunch.LocalBunch[:, bunch.BunchAttribute.Attribute['z']]

    x = r * np.cos(fi)
    y = r * np.sin(fi)

    # 提取 z=0 平面的电场和电势分布
    z_index = Ex_distribution.shape[2] // 2
    Ex_slice = Ex_distribution[:, :, z_index]
    Ey_slice = Ey_distribution[:, :, z_index]
    charge_slice = charge_distribution[:, :, z_index]

    # 创建 X 和 Y 网格点
    xlabels = np.linspace(x.min(), x.max(), Ex_distribution.shape[0])
    ylabels = np.linspace(y.min(), y.max(), Ey_distribution.shape[1])
    Xmesh, Ymesh = np.meshgrid(xlabels, ylabels, indexing='ij')

    # 归一化电场矢量
    magnitude = np.max(np.sqrt(Ex_slice ** 2 + Ey_slice ** 2))
    Ex_normalized = Ex_slice / (magnitude + 1e-10)
    Ey_normalized = Ey_slice / (magnitude + 1e-10)

    # 绘制电势的等高线图
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(Xmesh, Ymesh, Ey_slice, levels=50, cmap='rainbow')
    plt.colorbar(contour)
    plt.quiver(Xmesh, Ymesh, Ex_normalized, Ey_normalized, scale=30, color='k')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Electric Potential and Field Distribution at z=0")
    plt.show()


if __name__ == "__main__":
    main()
