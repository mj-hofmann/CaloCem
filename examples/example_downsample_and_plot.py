#%%
from pathlib import Path
import matplotlib.pyplot as plt

import CaloCem.tacalorimetry as ta

parentfolder = Path(__file__).parent.parent
datapath = parentfolder / "CaloCem" / "DATA"
plotpath = parentfolder / "docs" / "assets"
testpath = parentfolder / "tests"

processparams = ta.ProcessingParameters()
processparams.downsample.apply = True
processparams.downsample.num_points = 200


# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*data_[1].csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    processparams=processparams,
)


#%%
fig, ax = plt.subplots()
for name, group in tam._data.groupby("sample_short"):
    ax.plot(group["time_s"]/3600, group["normalized_heat_flow_w_g"]*1000, "x-", label=name)
#ax.set_xlim(0,24)
# ax.set_ylim(0,5)
plt.show()

# %%
print(len(tam._data))
# %%
