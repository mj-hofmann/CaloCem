import matplotlib.pyplot as plt
import pandas as pd

import sys
from pathlib import Path
parentfolder = Path(__file__).cwd().parent
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    # regex="(.*csv$)|(Exp_[345].*)",
    regex=r"c3a.csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

tam.apply_tian_correction(300,1.5e-2)

fig, ax = plt.subplots()
ax.plot(tam._data["time_s"], tam._data["normalized_heat_flow_w_g"])
ax.plot(tam._data["time_s"], tam._data["normalized_heat_flow_w_g_tian"])
ax.set_xlim(0,1000)
plt.show()