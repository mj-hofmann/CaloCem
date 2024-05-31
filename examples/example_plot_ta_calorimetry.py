# %%

from TAInstCalorimetry import tacalorimetry
from pathlib import Path

datapath = Path(__file__).parent.parent / "TAInstCalorimetry" / "DATA"

# %% use class based approach

# experiments via class
tam = tacalorimetry.Measurement(
    folder=datapath,
    regex="myexp[1-2].csv",
    show_info=True,
    auto_clean=False,
)

# get sample and information
data = tam.get_data()
info = tam.get_information()


# %% basic plotting
tam.plot(y="heat_flow_w")
# show plot
tacalorimetry.plt.show()


# %% basic plotting
tam.plot(y="heat_j")
# show plot
tacalorimetry.plt.show()


# %% customized plotting

ax = tam.plot(
    y="normalized_heat_flow_w_g",
    t_unit="d",  # time axis in hours
    y_unit_milli=True,
    regex="2",  # regex expression for filtering
)

# set upper limits
ax.set_ylim(top=5)
ax.set_xlim(right=2)
tacalorimetry.plt.show()


#%% get table of cumulated heat at certain age

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
