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
    regex=r".*data_[1-3].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %% plot

processparams = ProcessingParameters()

# get peak onsets via alternative method
for name, group in tam._data.groupby("sample_short"):
    fig, ax = plt.subplots()
    onsets = tam.get_astm_c1679_characteristics(
        processparams=processparams,
        show_plot=True,
        individual=True,
        ax=ax,
        regex=f".*{name}.*",# ".*data_1.*",
        xunit="h",
        )
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.005)
    plt.show()

print(onsets)
# %%
