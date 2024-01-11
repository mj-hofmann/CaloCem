#%%
import sys
from pathlib import Path
import matplotlib.pyplot as plt

parentfolder = Path(__file__).cwd()
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r"myexp[1].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

# get peak onsets via alternative method
onsets = tam.get_astm_c1679_characteristics(cutoff_min=15)

print(onsets)
# %%
