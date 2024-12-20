import json
import time
import os
import numpy as np
from Cal_ISO import CheckISOBmapConfigNew


config_data = {}
###############################################################
# some general parameters
config = dict()
config['date'] = time.time()

###############################################################
# machine parameters
machine = dict()
machine['energy_inj'] = 3
machine['energy_ext'] = 150

###############################################################
# Bmap configuration
Bmap = dict()
Bmap['Type'] = 'SCALE'
# 可选：等时'ISOCHRONOUS', or 等比'SCALE'
Bmap['NSector'] = 10

Bmap['theta_step_rad'] = np.deg2rad(0.01)  # unit: rad
Bmap['rmin_max_step_m'] = (1.8, 4.0, 0.001)  # unit: m， 比实际Bmap范围大一些，避免迭代过程中超出Bmap范围

Bmap['interval_1'] = (1.0/3.0, 1.0/3.0, 1.0/3.0)  # unit: 1
Bmap['positive_or_negative_1'] = (0, 1, 0)  # unit: 1
Bmap['fringe_width_1'] = (0, 0.8, 0)  # unit: 1
Bmap['SpiralAngle_deg'] = 30  # unit: deg

Bmap['orbital_freq_MHz'] = 4.5  # unit: MHz, Type为等时'ISOCHRONOUS'时起效
Bmap['k_value'] = 5.0  # unit: 1, Type为等比'SCALE'时起效
Bmap['B0_max_T'] = 1.7  # unit: T, 参考半径处的B0, Type为等比'SCALE'时起效

###############################################################
###############################################################
# Bz configuration, not defined by the user
BzInfo = dict()
BzInfo['IsoCoef'] = None  # not defined by the user
BzInfo['IsoError'] = None  # not defined by the user
Bmap['B0_T'] = Bmap['B0_max_T'] / (Bmap['rmin_max_step_m'][1]**Bmap['k_value'])  # not defined by the user

# ready to write out
config_data['config'] = config
config_data['machine'] = machine
config_data['Bmap'] = Bmap
config_data['BzInfo'] = BzInfo

if __name__ == "__main__":
    coefficients_tupel = CheckISOBmapConfigNew(config_data)
    config_data['Bmap']['coefficients_tupel'] = coefficients_tupel
    current_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".json"
    with open(current_file_name, 'w') as f_out:
        json.dump(config_data, f_out, indent=2, sort_keys=True)

