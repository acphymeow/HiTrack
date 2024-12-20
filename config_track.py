import os
import json
import time
import numpy as np

cf = {}
###############################################################################
# 通用参数
config = dict()
config['date'] = time.time()  # 当前时间戳

###############################################################################
# 追踪参数
track = dict()
track['start_EkMeV'] = 3.0  # 初始动能 (MeV)
track['start_azimuth'] = 0.0  # 注入点方位角
stop_condition = {'max_stepsN': 15000,
                  'max_turn': 12000.5,
                  'max_time': float('inf')}
step_condition = {'step_fi': 0.10, 'max_step_fi': 0.08, 'min_step_fi': 0.13}  # deg
track['start_azimuth'] = 0.0  # 注入点方位角
track['stop_condition'] = stop_condition
track['step_condition'] = step_condition
track['enable_SC'] = True

###############################################################################
# check模式
check = dict()
check['CheckBunch'] = True
check['PlotRMS'] = (4, 4, 4)

###############################################################################
# 磁场和SEO配置
BmapAndSEO = dict()
BmapAndSEO['SEO'] = './resultsSEO/Bmap-2024-12-05-161711'  # 磁场和SEO文件路径
BmapAndSEO['max_order'] = 2

###############################################################################
# 电场配置
Emap = dict()
Emap['maps'] = './emaps'  # 电场文件路径
Emap['voltage_kV'] = 50   # 电场强度, kV

###############################################################################
# 注入小束团参数
BunchPara = dict()
BunchPara['ParticleDensity'] = 3.0e7  # 每个宏粒子对应的实际粒子数
BunchPara['ParticleNum'] = [1000, ]  # 每个子束团的宏粒子数
BunchPara['TransverseREmit'] = [1, ]  # 横向发射度 (pi*mm*mrad)
BunchPara['TransverseZEmit'] = [1, ]  # 横向发射度 (pi*mm*mrad)
BunchPara['LongitudeT'] = [100, ]  # 纵向长度 (ns)
BunchPara['LongitudeDEk'] = [0.004, ]  # 纵向能散 (MeV)
BunchPara['InjTimeNanoSec'] = [0, ]  # 注入时间范围 (ns)
BunchPara['RF0'] = np.deg2rad(0.021 * 360.0)  # RF 相位 (rad)
BunchPara['TransverseDistType'] = 'gauss'  # 横向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag')
BunchPara['LongitudeDistType'] = 'gauss'  # 纵向分布类型 (可选类型'gauss', 'kv', 'waterbag', 'hollow_waterbag')

# 涂抹
Paint = dict()
Paint['BunchNum'] = 1000  # 注入子束团个数
Paint['TimeInterval'] = 100.0  # 时间间隔ns
Paint['Curve'] = './PaintCurve/Curve1.paint'

###############################################################################
# Dump 配置
DumpPara = dict()
# StepDump
StepDump = {
    "type": "StepDump",  # 按步数间隔保存粒子 每个粒子单独保存为一个文件 按ID区分
    "step_interval": 50,  # 保存间隔的步数
    "num_particles_to_dump_global": 100,  # 全局保存的粒子数
    "save_folder": "./dump5/step_dump_sc_1"  # 保存路径
}
# PositionDump
PositionDump = {
    "type": "PositionDump",  # 粒子经过特定位置时保存
    "dump_azimuth": 10,  # 保存间隔的方位角 (度)
    "save_folder": "./dump5/step_dump_sc_3"  # 保存路径
}
# StepDumpBunch
StepDumpBunch = {
    "type": "StepDumpBunch",  # 按步数间隔保存整个粒子束 按步数保存bunch为一个单独文件 按步数区分
    "step_interval": 500,  # 保存间隔的步数
    "save_folder": "./dump5/step_dump_sc_2"  # 保存路径
}
# 将所有 dump 配置组合到模块中
DumpPara['modules'] = [StepDump, PositionDump, StepDumpBunch]

###############################################################################
# 整体配置
cf['config'] = config
cf['track'] = track
cf['BmapAndSEO'] = BmapAndSEO
cf['BunchPara'] = BunchPara
cf['DumpPara'] = DumpPara
cf['check'] = check
cf['Emap'] = Emap

# 保存 JSON 配置文件
if __name__ == "__main__":
    current_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".json"
    with open(current_file_name, 'w') as f_out:
        json.dump(cf, f_out, indent=2, sort_keys=True)
    print(f"配置文件已保存为 {current_file_name}")

