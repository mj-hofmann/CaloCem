import sys
from pathlib import Path
import itertools

parentfolder = Path(__file__).cwd().parent
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*exp[48].csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)


#%%
# # plot

ycols = ["normalized_heat_flow_w_g", "normalized_heat_j_g"]
limits = [1, 30]
combinations = list(itertools.product(ycols, limits))

fig, axs = ta.plt.subplots(2, 2, layout="constrained")
for ax, (col, lim) in zip(axs.flatten(), combinations):
    tam.plot(y=col, t_unit="h", ax=ax)
    ax.set_xlim(0, lim)
    ax.get_legend().remove()
ta.plt.show()



# %%
