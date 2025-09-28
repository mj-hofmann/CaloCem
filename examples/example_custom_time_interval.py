# %%

from pathlib import Path

from calocem.tacalorimetry import Measurement

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


#%%
heat = tam.get_cumulated_heat_at_hours(target_h = 72, cutoff_min = 4500/60)
# %%
