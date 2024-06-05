#%%
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import TAInstCalorimetry.tacalorimetry as ta

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r"myexp[1].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

processparams = ta.ProcessingParameters()

# get peak onsets via alternative method
onsets = tam.get_astm_c1679_characteristics(processparams=processparams)

print(onsets)
# %%
