
#%%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem.tacalorimetry import Measurement
from calocem.processparams import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r"calorimetry_data_[12].*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 60

heats = tam.get_cumulated_heat_at_hours(target_h = 24, processparams=processparams)
heats_depracated = tam.get_cumulated_heat_at_hours(cutoff_min=30, target_h=24)
# %%
