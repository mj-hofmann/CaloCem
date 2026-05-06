# %%
from pathlib import Path

import matplotlib.pyplot as plt

from calocem import Measurement, ProcessingParameters

datapath = Path(__file__).parent.parent / "calocem" / "DATA"

# %% load data
tam = Measurement(
    folder=datapath,
    regex="calorimetry_data_[7].csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

# %% configure gradient processing
processparams = ProcessingParameters()
processparams.rolling_mean.apply = True
processparams.rolling_mean.window = 11
processparams.spline_interpolation.apply = True
processparams.spline_interpolation.smoothing_1st_deriv = 1e-14
processparams.cutoff.cutoff_min = 75
processparams.plotting.figsize = (5, 4)

# %% plot heat flow with gradient on secondary y-axis
ax, ax_grad = tam.plot_heatflow_with_gradient(
    processparams=processparams,
    t_unit="h",
    y_unit_milli=True,
    gradient_unit_milli=True,
    show_zero_line=True,
)

ax.set_xlim(0, 48)
plt.tight_layout()
plt.show()

# %%
