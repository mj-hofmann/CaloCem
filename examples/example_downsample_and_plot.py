#%%
from pathlib import Path
import matplotlib.pyplot as plt

#import CaloCem.tacalorimetry as ta
from calocem import Measurement
from calocem import ProcessingParameters

parentfolder = Path(__file__).parent.parent
datapath = parentfolder / "calocem" / "DATA"
plotpath = parentfolder / "docs" / "assets"
testpath = parentfolder / "tests"

processparams = ProcessingParameters()
processparams.downsample.apply = True
processparams.downsample.num_points = 400
# processparams.downsample.section_split = True
# processparams.downsample.section_split_time_s = 3600
processparams.downsample.baseline_weight = 0.1
processparams.cutoff.cutoff_min = 70
processparams.cutoff.cutoff_max = 55 * 60

# experiments via class
tam_d = Measurement(
    folder=datapath,
    regex=r".*data_[1].csv",
    #regex="JAA.*",
    # regex="downsample.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    processparams=processparams,
)

tam = Measurement(
    folder=datapath,
    regex=r".*data_[1].csv",
    # regex="downsample.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


# %%

fig, ax = plt.subplots()
for (name, group), (name2, group2) in zip(tam.get_data().groupby("sample_short"), tam_d.get_data().groupby("sample_short")):
    ax.plot(group["time_s"]/3600, group["normalized_heat_flow_w_g"]*1000, "-", label=name)
    ax.plot(group2["time_s"]/3600, group2["normalized_heat_flow_w_g"]*1000, "-", label=name2 + "_downsampled")
ax.legend()
ax.set_xlim(0,60)
ax.set_ylim(0,4)

# %%


