# 程序使用手册

## 1. 引言

### 1.1 简介
- 这是一个用于模拟粒子加速器中粒子运动的程序。
- 程序使用3维磁场map作为输入文件，利用4阶Runge-Kutta方法求解洛伦兹力方程，得到粒子的运动信息。
- 本程序利用FFT和iFFT加速的PIC方法计算3维空间电荷效应。
- 包含涂抹注入（开发中）。

### 1.2 依赖库
  - Python 3.12+
  - openMPI
  - NumPy - (python package)
  - SciPy - (python package)
  - Numba - (python package)
  - mpi4py - (python package)
---

## 2. 安装与配置

### 2.1 安装步骤

直接下载代码，安装依赖即可。
#### Windows 安装
1. 下载并解压程序压缩包。
2. 打开命令行终端，进入解压后的目录。
3. 运行以下命令：
   ```bash
   pip install -r requirements.txt
   ```

#### Linux 安装

1. 下载程序源码。
2. 安装所需依赖：
    ```bash
    pip install -r requirements.txt
    ```


## 3. 程序使用
进行模拟时，主要分为三步：

1. 生成 Bmap 文件。
2. 计算闭轨。
3. 借助于闭轨信息，注入粒子，进行多粒子模拟。

### 3.1 生成等比型FFAG Bmap 文件

#### 3.1.1 编写config.py文件
和PyORBIT类似，Bmap文件参数在config.py文件中定义。生成Bmap的config.py文件需要包含以下字典。

首先是config，包含时间戳，作者等标识，不参与计算。
```python
# some general parameters
config = dict()
config['date'] = time.time()
```

machine为机器参数，目前只包含注入引出能量两个参数，单位MeV。
```python
# machine parameters
machine = dict()
machine['energy_inj'] = 300
machine['energy_ext'] = 600
```

Bmap包含所有lattice参数，包括lattice类型，排列，场强等：
```python
# Bmap configuration
Bmap = dict()
Bmap['Type'] = 'SCALE'
# 可选：等时'ISOCHRONOUS', or 等比'SCALE'
Bmap['NSector'] = 12

Bmap['theta_step_rad'] = np.deg2rad(0.01)  # unit: rad
Bmap['rmin_max_step_m'] = (9.0, 10.0, 0.001)  # unit: m

Bmap['interval_1'] = (0.35, 0.3, 0.35)  # unit: 1
Bmap['positive_or_negative_1'] = (0, 1, 0)  # unit: 1
Bmap['fringe_width_1'] = (0, 0.5, 0)  # unit: 1
Bmap['SpiralAngle_deg'] = 45  # unit: deg

Bmap['orbital_freq_MHz'] = 4.5  # unit: MHz, Type为等时'ISOCHRONOUS'时起效
Bmap['k_value'] = 6.0  # unit: 1, Type为等比'SCALE'时起效
Bmap['B0_max_T'] = 1.8  # unit: T, 参考半径处的B0, Type为等比'SCALE'时起效
```
以下是具体解释：

**Bmap['Type']：** 定义磁场map类型，可选等时'ISOCHRONOUS'或等比'SCALE'

对于 **等比型**，磁场径向分布按照下式计算：

$$
B(r) = B_0 \left(\frac{r}{r_0}\right)^k
$$

其中，$B_0$ 是参考半径处的磁场强度，$r_0$ 是参考半径，$k$ 是比例因子。

对于 **等时型**，磁场径向分布按照以下多项式形式计算：

$$
B(r) = a_0 + a_1 (r - r_{\text{min}}) + a_2 (r - r_{\text{min}})^2 + a_3 (r - r_{\text{min}})^3 + \cdots
$$

其中，$r_{\text{min}}$ 是起始半径，$a_0, a_1, a_2, \dots$ 是多项式的系数。这些系数需要用户给定目标频率，由程序自动搜索。

**Bmap['NSector']:** Lattice 的周期数，表示磁场的周期数。本例中，Lattice 有 12 个周期。

**Bmap['theta_step_rad']:** 方位角步长，控制磁场映射的步长。单位为 rad，本例中步长为 0.01°（已转换为 rad）。

**Bmap['rmin_max_step_m']:** 磁场映射的起始半径、终止半径和半径步长。单位为 m。本例中的半径范围从 9.0 m 到 10.0 m，步长为 0.001 m。

**Bmap['interval_1'] = (0.35, 0.3, 0.35)**，

**Bmap['positive_or_negative_1'] = (0, 1, 0)**，

**Bmap['fringe_width_1'] = (0, 0.5, 0)**

**Bmap['SpiralAngle_deg'] = 45**

这几个参数表示一个周期内磁铁的排列方式，具体来说

interval_1表示一个周期分为几段，以及每段的角宽度占比。
本例中，一个周期分为3段，第1段占35%，第2段占30%，第3段占35%。

positive_or_negative_1表示每段的极性，1表示放置一个正磁极，0表示放置一个漂移段，负数表示放置一个负磁极，数值为相对正磁极的强度。
本例中，第1段为漂移段，第2段为正磁极，第3段为漂移段。

fringe_width_1表示磁极边缘场宽度相对磁极宽度的比例。本例中，磁极边缘场宽度为磁极平顶段宽度的50%

SpiralAngle_deg表示磁极的螺旋角


**Bmap['orbital_freq_MHz'] = 4.5：** 表示等时场的目标回旋频率，这里是4.5MHZ，只在选择等时型磁场时生效。

**Bmap['k_value'] = 6.0：** 等比型磁场时生效，k值

**Bmap['B0_max_T'] = 1.8：** 等比型磁场时生效，参考半径（最大半径）处的磁场


#### 3.1.2 生成Lattice参数， 检查参数是否合理

在config.py所在目录执行
```bash
python config_track.py
```
随后生成config.json文件，包含lattice参数信息。屏幕上也会打印一些磁场信息
```
Bmap type = SCALE
****************************************************************************************************
Bmap rmin = 9.000m, rmax = 10.000m
Ek = 285.825 MeV for R9.020m orbit, Ek = 912.941 MeV for R9.980m orbit.
初始化完成，读取 2 阶的展开系数矩阵
****************************************************************************************************
检查能量范围:
注入能量 (energy_inj): 300 MeV
引出能量 (energy_ext): 600 MeV
BMapData 的能量范围: [285.82548269733866, 912.9412956933671] MeV
能量范围在 BMapData 的有效范围内。

```
用户可以检查Bmap的能量范围是否符合要求，否则可以调整磁场参数使其匹配。


#### 3.1.3 生成Bmap
利用刚才生成的.json格式的lattice参数文件，执行以下命令即可生成Bmap
```bash
python ./main_GenerateBmaps.py -j ./configII.json -f ./test_coef -m 3
```
这个命令包含-j, -f, -m 等3个参数

**-j ./configII.json** 输入的json文件

**-f ./test_coef** 保存生成的Bmap文件到该文件夹

**-m 3** 表示最多考虑2*3+1=7阶非线性项。m大于4时，计算速度会很慢，最多用4即可。


#### 3.1.4 计算闭轨和光学参数
进行多粒子模拟前，还需要计算闭轨和光学参数，因为注入和跟踪过程中需要读取一些闭轨参数作为参考。
打开main_SEO.py文件，输入能量点和Bmap路径
```python
# input parameters given by user
EkRange = np.array([350, 400, 450, 500, 550])
BMapData = FFAG_BField_new('./coeff_matrices_2/', 0, flag3D=True)
```
再执行
```
python main_SEO.py
```
进行闭轨和光学参数计算

闭轨数据计算结果会保存在根目录下./resultsSEO/文件夹中

#### 3.1.5 多粒子跟踪
和PyORBIT一样，多粒子跟踪所需的参数都在config_track.py中定义。

一个典型的config_track.py文件包含以下内容：
首先是通用参数，包含时间戳和作者等信息，不参与计算。
```python
# 通用参数
config = dict()
config['date'] = time.time()  # 当前时间戳
```

其次是跟踪参数，目前只有起始能量一个参数
```python
# 追踪参数
track = dict()
track['start_EkMeV'] = 300.0  # 初始动能 (MeV)
```

check模式是用于检查输入参数，包括绘制初始bunch分布，初始Bmap和Emap，闭轨信息等。

开启check模式时，程序不进行跟踪，画图后直接终止程序。跟踪时需要关掉check模式。
```python
# check模式
check = dict()
check['enable'] = True
check['CheckBunch'] = True
check['PlotRMS'] = (4, 4, 4)
check['CheckBMap'] = True
check['CheckEMap'] = True
check['CheckSEO'] = True
```

磁场信息和SEO是匹配的，输出闭轨信息时，已经将磁场路径包含在SEO文件中，所以这里只要输入SEO文件路径，即可同时读取SEO和磁场。

max_order和非线性项的阶数相关，计算的非线性项阶数为2*max_order+1阶。max_order=3的话，计算的最高阶数为7阶，max_order=4的话，计算的最高阶数为9阶。
```python
# 磁场和SEO配置
BmapAndSEO = dict()
BmapAndSEO['SEO'] = './resultsSEO/Bmap-2024-11-02-020425'  # 磁场和SEO文件路径
BmapAndSEO['max_order'] = 3
```


Emap用于读取归一化的电场map，电压，RF曲线等
```python
# 电场配置
Emap = dict()
Emap['maps'] = './emaps'  # 电场文件路径
Emap['voltage_kV'] = 50   # 电场强度, kV
```

BunchPara为初始束团的粒子数，发射度，6维相空间分布类型等参数。

BunchPara['ParticleNum'] = [1000, ] 表示注入1个小束团，包含1000个粒子

可以注入多个束团，例如BunchPara['ParticleNum'] = [1000, 2000] 表示注入2个小束团，包含1000个粒子和2000个粒子

BunchPara['InjTimeNanoSec'] = [-1, ]  表示束团的注入时刻，跟踪从0s开始，-1表示跟踪时已注入。正数表示实际的注入时刻（ns）

注入多个束团时，需要分别指定注入时刻，例如BunchPara['InjTimeNanoSec'] = [-1, 10,]

BunchPara['TransverseREmit'], 

BunchPara['TransverseZEmit'], 

BunchPara['LongitudeFi'], 

BunchPara['LongitudeDEk']

分别表示每个小束团的R方向横向发射度, Z方向横向发射度, 纵向长度(角度值，deg)，纵向能散。注入多个束团时，需要分别指定相应数值。

BunchPara['TransverseDistType'],

BunchPara['LongitudeDistType']

分别表示横向和纵向分布类型，可选择'gauss', 'kv', 'waterbag', 'hollow_waterbag'4种。
```python
# 粒子束参数
BunchPara = dict()
BunchPara['ParticleDensity'] = 1e2  # 每个宏粒子对应的实际粒子数
BunchPara['ParticleNum'] = [1000, ]  # 每个子束团的宏粒子数
BunchPara['TransverseREmit'] = [10, ]  # 横向发射度 (pi*mm*mrad)
BunchPara['TransverseZEmit'] = [10, ]  # 横向发射度 (pi*mm*mrad)
BunchPara['LongitudeFi'] = [2, ]  # 纵向长度 (deg)
BunchPara['LongitudeDEk'] = [1, ]  # 纵向能散 (MeV)
BunchPara['InjTimeNanoSec'] = [-1, ]  # 注入时间范围 (ns)
BunchPara['RF0'] = np.deg2rad(0.021 * 360.0)  # RF 相位 (rad)
BunchPara['TransverseDistType'] = 'gauss'  # 横向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag')
BunchPara['LongitudeDistType'] = 'gauss'  # 纵向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag')

```

DumpPara用于在跟踪过程中保存粒子信息。提供了3种保存方法
1. 按粒子ID保存，每个粒子对应一个单独文件，每隔指定的步数保存一次，适用于绘制粒子轨迹。
2. 按Bunch保存，每隔指定的步数输出一次Bunch内所有粒子坐标信息。
3. 按Bunch保存，数据记录了粒子穿越特定方位角时的状态。

| Dump 类型         | 文件粒度             | 数据组织                | 适用场景                  |
|--------------------|----------------------|-------------------------|---------------------------|
| **StepDump**       | 每个粒子一个文件     | 按时间存储粒子状态      | 个体粒子轨迹分析          |
| **StepDumpBunch**  | 每个时间步一个文件   | 按粒子存储全局状态      | 全局状态、时间片段分析    |
| **PositionDump**   | 每个方位角一个文件   | 按粒子存储穿越点信息    | 特定方位角粒子剖面分析    |

```python
# Dump 配置
DumpPara = dict()
# StepDump
StepDump = {
    "type": "StepDump",  # 按步数间隔保存粒子 每个粒子单独保存为一个文件 按ID区分
    "step_interval": 50,  # 保存间隔的步数
    "num_particles_to_dump_global": 100,  # 全局保存的粒子数
    "save_folder": "./step_dump_sc_4"  # 保存路径
}
# PositionDump
PositionDump = {
    "type": "PositionDump",  # 粒子经过特定位置时保存
    "azimuth_interval": 10,  # 保存间隔的方位角 (度)
    "save_folder": "./step_dump_sc_2"  # 保存路径
}
# StepDumpBunch
StepDumpBunch = {
    "type": "StepDumpBunch",  # 按步数间隔保存整个粒子束 按步数保存bunch为一个单独文件 按步数区分
    "step_interval": 500,  # 保存间隔的步数
    "save_folder": "./step_dump_sc_3"  # 保存路径
}
# 将所有 dump 配置组合到模块中
DumpPara['modules'] = [StepDump, PositionDump, StepDumpBunch]

```




#### 3.1.2 生成等时型FFAG Bmap 文件


    ```bash
    python main.py --config config.ini
	```


## 4. 更新记录
- **v1.0.0** (2024-11-15)：首次发布。

