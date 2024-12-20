import argparse
from FFAG_track import FFAG_RungeKutta, FFAG_SearchSEO
from FFAG_Field import FFAG_BField_new
from FFAG_ParasAndConversion import FFAG_GlobalParameters
from FFAG_BetaFunc import FFAG_BetaFuncCalc
import numpy as np

def main(start_energy, end_energy, num_points, coeff_matrices_path):
    # Generate energy range
    EkRange = np.linspace(start_energy, end_energy, num_points, endpoint=True)

    # Load BMap data
    BMapData = FFAG_BField_new(coeff_matrices_path, 1, flag3D=True)

    # Extend field
    BMapData.ExtendField(100, 100, 5)

    # Generate global parameters
    GlobalParas = FFAG_GlobalParameters()
    GlobalParas.AddBMap(BMapData)

    # Search SEO
    SEO_FilePath = FFAG_SearchSEO(GlobalParas).SearchSEOsControllerVect(EkRange)

    # Calculate beta functions
    FFAG_BetaFuncCalc().CalcBetaFunc(SEO_FilePath, EkRange)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="FFAG Simulation Command-Line Tool")
    parser.add_argument("-se", "--start_energy", type=float, required=True, help="起始能量 (单位 MeV)")
    parser.add_argument("-ee", "--end_energy", type=float, required=True, help="终止能量 (单位 MeV)")
    parser.add_argument("-ne", "--num_points", type=int, required=True, help="能量点个数（必须大于 5）")
    parser.add_argument("-bp", "--coeff_matrices_path", type=str, required=True, help="磁场系数矩阵路径")

    # Parse arguments
    args = parser.parse_args()

    # Check if num_points is greater than 5
    if args.num_points <= 5:
        raise ValueError("能量点个数必须大于 5，请重新设置 -ne 参数。")

    # Run the main function with parsed arguments
    main(args.start_energy, args.end_energy, args.num_points, args.coeff_matrices_path)
