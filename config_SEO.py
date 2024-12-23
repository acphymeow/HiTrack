import json
import time
import os
import numpy as np
from Cal_ISO import CheckISOBmapConfigNew


config_data = {}
###############################################################
# some general parameters
config = dict()
config['start_Ek'] = 10.0
config['end_Ek'] = 150.0
config['delta_Ek'] = 10.0
config['extra_Ek'] = (3.0, 5.0)
config['Bmap_path'] = '.\\Bmap'

# ready to write out
config_data['config'] = config

if __name__ == "__main__":
    coefficients_tupel = CheckISOBmapConfigNew(config_data)
    config_data['Bmap']['coefficients_tupel'] = coefficients_tupel
    current_file_name = os.path.splitext(os.path.basename(__file__))[0] + ".json"
    with open(current_file_name, 'w') as f_out:
        json.dump(config_data, f_out, indent=2, sort_keys=True)

