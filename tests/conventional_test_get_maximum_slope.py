import sys
from pathlib import Path

parentfolder = Path(__file__).cwd()
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r"myexp[1-4]",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

# get peak onsets via alternative method
onsets = tam.get_maximum_slope(show_plot=True)

print(onsets)
# %%
