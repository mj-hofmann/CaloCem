import sys
from pathlib import Path

parentfolder = Path(__file__).cwd().parent
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r"(c3a)|(opc_3).csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

fig, axs = ta.plt.subplots(2, 2, layout="constrained")

# plot
ax = tam.plot(t_unit="min", y_unit_milli=False, ax=axs[0, 1])
# set limit
ax.set_xlim(0, 30)
ax.set_ylim(0, 0.3)

# plot
ax = tam.plot(t_unit="h", y="normalized_heat_j_g", y_unit_milli=False, ax=axs[1, 0])
# set limit
ax.set_xlim(0, 48)
ax.set_ylim(0, 750)
# remove legend (set as default)
ax.get_legend().remove()
