# %%

from pathlib import Path

from calocem import Measurement
from calocem import ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"
plotpath = Path(__file__).parent.parent / "docs" / "assets"

# %% use class based approach

# experiments via class
tam = Measurement(
    folder=datapath,
    regex="calorimetry_data_[1].csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

# get sample and information
data = tam.get_data()
info = tam.get_information()

processparams = ProcessingParameters()
processparams.cutoff.cutoff_min = 75 

#%%
heat = tam.get_cumulated_heat_at_hours( processparams, target_h=4)
# %%
