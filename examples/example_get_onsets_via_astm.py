#%%
import sys
from pathlib import Path
import matplotlib.pyplot as plt

from calocem.tacalorimetry import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r".*data_4.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

processparams = ProcessingParameters()

# get peak onsets via alternative method
onsets = tam.get_astm_c1679_characteristics(processparams=processparams)

print(onsets)
# %%
