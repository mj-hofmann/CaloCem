# %%

from TAInstCalorimetry import tacalorimetry
from pathlib import Path

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"
plotpath = Path(__file__).parent.parent / "docs" / "assets"

# %% use class based approach

# experiments via class
tam = tacalorimetry.Measurement(
    folder=datapath,
    regex="calorimetry_data_[1].csv",
    show_info=True,
    auto_clean=False,
)

# get sample and information
data = tam.get_data()
info = tam.get_information()


# %% basic plotting
tam.plot()
# save plot
tacalorimetry.plt.savefig(plotpath / "basic_plot.png")


# %% basic plotting
tam.plot(y="normalized_heat_j_g")
# show plot
tacalorimetry.plt.show()


# %% customized plotting

ax = tam.plot(
    y="normalized_heat_flow_w_g",
    t_unit="h",  # time axis in hours
    y_unit_milli=True,
)

# set upper limits
ax.set_ylim(0, 6)
ax.set_xlim(0, 48)
ax.legend(bbox_to_anchor=(1., 1), loc="upper right")
tacalorimetry.plt.show()


# %% get table of cumulated heat at certain age

# define target time
target_h = 5

# get cumlated heat flows for each sample
cum_h = tam.get_cumulated_heat_at_hours(target_h=target_h, cutoff_min=1)
print(cum_h)

# show cumulated heat plot
ax = tam.plot(t_unit="h", y="normalized_heat_j_g", y_unit_milli=False)

# guide to the eye line
ax.axvline(target_h, color="gray", alpha=0.5, linestyle=":")
ax.set_ylim(top=100)
ax.set_xlim(right=12)
tacalorimetry.plt.show()


# %%
