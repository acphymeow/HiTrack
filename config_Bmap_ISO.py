import json
import time
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
machine['energy_inj'] = 400
machine['energy_ext'] = 500

###############################################################
# Bmap configuration
Bmap = dict()
Bmap['Type'] = 'ISOCHRONOUS'
# 可选：等时'ISOCHRONOUS', or 等比'SCALE'
Bmap['NSector'] = 10

Bmap['theta_step_rad'] = np.deg2rad(0.01)  # unit: rad
Bmap['rmin_max_step_m'] = (6.1, 9.2, 0.001)  # unit: m

Bmap['interval_1'] = (0.3, 0.4, 0.3)  # unit: 1
Bmap['positive_or_negative_1'] = (0, 1, 0)  # unit: 1
Bmap['fringe_width_1'] = (0, 0.5, 0)  # unit: 1
Bmap['SpiralAngle_deg'] = 45  # unit: deg

Bmap['orbital_freq_MHz'] = 4.5  # unit: MHz
Bmap['k_value'] = 6.0  # unit: 1
Bmap['B0_T'] = 1.8 / ((10.0)**Bmap['k_value'])  # unit: T/m**k


###############################################################
###############################################################
# Bz configuration, not defined by the user
BzInfo = dict()
BzInfo['IsoCoef'] = None
BzInfo['IsoError'] = None

# ready to write out
config_data['config'] = config
config_data['machine'] = machine
config_data['Bmap'] = Bmap
config_data['BzInfo'] = BzInfo

if __name__ == "__main__":
    coefficients_tupel = CheckISOBmapConfigNew(config_data)
    config_data['Bmap']['coefficients_tupel'] = coefficients_tupel
    with open('configIII.json', 'w') as f_out:
        json.dump(config_data, f_out, indent=2, sort_keys=True)

