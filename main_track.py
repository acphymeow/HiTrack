import argparse
import os
import json
import numpy as np
from distribution_generators import GenerateBunches, LoadSEOParams
from FFAG_track import FFAG_RungeKutta, FunctionForAccelerationBunch_dt
from FFAG_ParasAndConversion import FFAG_GlobalParameters
from FFAG_Field import FFAG_BField_new, FFAG_EField
from FFAG_Bunch import FFAG_Bunch
from FFAG_dump import StepDump, StepDumpBunch, PositionDump, Dumps

# 从 JSON 文件读取配置
def load_config_from_json(json_path):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return config

# 主函数
def main(config_file):
    # 读取 JSON 配置
    config = load_config_from_json(config_file)

    # 从配置中获取参数
    SEOPATHName = config['BmapAndSEO']['SEO']
    EmapPATHName = config['Emap']['maps']
    Ek_MeV = config['track']['start_EkMeV']
    Azimuth_start = config['track']['start_azimuth']
    check = config['check']

    # 粒子束参数
    BunchPara = config['BunchPara']

    # 将 check 中的键值映射到 BunchPara 中
    BunchPara['PlotFlag'] = check['CheckBunch']  # 对应 JSON 中的 check['CheckBunch']
    BunchPara['PlotRMS'] = check['PlotRMS']     # 对应 JSON 中的 check['PlotRMS']
    BunchPara['SEO'] = SEOPATHName
    BunchPara["InjectEk"] = Ek_MeV
    BunchPara["InjPosition"] = Azimuth_start

    # 生成粒子束坐标分布
    array_dist_v = GenerateBunches(BunchPara)

    # 创建 FFAG_Bunch 对象并注入粒子
    BunchObj = FFAG_Bunch(array_dist_v, marcosize=BunchPara['ParticleDensity'])

    # StepDump 配置
    dumps_manager = Dumps()
    for dump_config in config['DumpPara']['modules']:
        if dump_config['type'] == 'StepDump':
            step_dump = StepDump(
                dump_config['step_interval'],
                num_particles_to_dump_global=dump_config['num_particles_to_dump_global'],
                save_folder=dump_config['save_folder']
            )
            dumps_manager.add_dump(step_dump)
        elif dump_config['type'] == 'PositionDump':
            position_dump = PositionDump(
                dump_config['dump_azimuth'],
                save_folder=dump_config['save_folder']
            )
            dumps_manager.add_dump(position_dump)
        elif dump_config['type'] == 'StepDumpBunch':
            bunch_dump = StepDumpBunch(
                dump_config['step_interval'],
                save_folder=dump_config['save_folder']
            )
            dumps_manager.add_dump(bunch_dump)

    # 加载 SEO 参数
    _, BmapPath, SEO_info = LoadSEOParams(os.path.join(SEOPATHName, "SEO_ini.txt"), Ek_MeV)
    GlobalParameters = FFAG_GlobalParameters()
    BmapPath = os.path.dirname(BmapPath)
    BMap = FFAG_BField_new(BmapPath, config['BmapAndSEO']['max_order'], flag3D=True)
    EMap = FFAG_EField(EmapPATHName)
    EMap.AddFreqEkCurve(SEO_info)
    EMap.AddVoltageValue(config['Emap']['voltage_kV'] * 1e3)  # 转换为 V
    GlobalParameters.AddBMap(BMap)
    GlobalParameters.AddEMap(EMap)
    GlobalParameters.AddSEOInfo(SEO_info)

    # 初始化追踪参数
    t_start = 0.0
    LocalParameters = dict()
    LocalParameters['stop_condition'] = config['track']['stop_condition']
    LocalParameters['step_condition'] = config['track']['step_condition']
    LocalParameters['step_dumps'] = dumps_manager
    LocalParameters['enable_SC'] = config['track']['enable_SC']

    # 创建 Runge-Kutta 求解器并运行追踪
    rk = FFAG_RungeKutta()
    rk.rk4_solve_dt_bunch3_withSC(FunctionForAccelerationBunch_dt,
                                  t_start, BunchObj,
                                  GlobalParameters, LocalParameters)

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="FFAG Simulation Tool")
    parser.add_argument(
        "-cf",
        type=str,
        required=True,
        help="Path to the JSON configuration file (e.g., config.json)"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(args.cf)

