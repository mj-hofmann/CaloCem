from pathlib import Path
import itertools

import CaloCem.tacalorimetry as ta

parentfolder = Path(__file__).parent.parent
datapath = parentfolder / "CaloCem" / "DATA"
plotpath = parentfolder / "docs" / "assets"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*bm.*.csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


#%% # plot

ycols = ["normalized_heat_flow_w_g", "normalized_heat_j_g"]
xlimits = [1, 48]
ylimits = [0.05, 0.005, 30, 300]
combinations = list(itertools.product(ycols, xlimits))

fig, axs = ta.plt.subplots(2, 2, layout="constrained")
for ax, (col, xlim), ylim in zip(axs.flatten(), combinations, ylimits):
    tam.plot(y=col, t_unit="h", y_unit_milli=False, ax=ax)
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.get_legend().remove()
ta.plt.savefig(plotpath / "subplot_example.png")



# %%

# Define subplot configurations more explicitly
plot_configs = [
    {"ycol": "normalized_heat_flow_w_g", "xlim": 1, "ylim": 0.05},
    {"ycol": "normalized_heat_flow_w_g", "xlim": 48, "ylim": 0.005},
    {"ycol": "normalized_heat_j_g", "xlim": 1, "ylim": 30},
    {"ycol": "normalized_heat_j_g", "xlim": 48, "ylim": 300},
]

fig, axs = ta.plt.subplots(2, 2, layout="constrained")
for ax, config in zip(axs.flatten(), plot_configs):
    tam.plot(y=config["ycol"], t_unit="h", y_unit_milli=False, ax=ax)
    ax.set_xlim(0, config["xlim"])
    ax.set_ylim(0, config["ylim"])
    ax.get_legend().remove()
ta.plt.show()
# %%
