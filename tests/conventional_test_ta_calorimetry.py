# %%
import os
import sys

from TAInstCalorimetry import tacalorimetry

# get path of script and set it as current path
pathname = os.path.dirname(sys.argv[0])
os.chdir(pathname)
filename = os.path.basename(__file__).replace(".py", "")

os.sys.path.append(pathname + os.sep + os.pardir + os.sep + "src")

# import

# %% use class based approach

# define data path
path_to_data = pathname + os.sep + os.pardir + os.sep + "DATA"

# experiments via class
tam = tacalorimetry.Measurement(
    folder=path_to_data,
    # regex="(.*csv$)|(Exp_[345].*)",
    show_info=True,
    auto_clean=True,
)

# get sample and information
data = tam.get_data()
info = tam.get_information()

# %% get samples
#

for sample, sample_data in tam.iter_samples():
    # print
    print(os.path.basename(sample))


# clean
del sample, sample_data


# %% basic plotting
#

tam.plot()
# show plot
tacalorimetry.plt.show()

# %% customized plotting

ax = tam.plot(
    t_unit="d",  # time axis in hours
    y_unit_milli=True,
    regex="3",  # regex expression for filtering
)

# set upper limits
ax.set_ylim(top=5)
ax.set_xlim(right=4)
# show plot
tacalorimetry.plt.show()


# %% get table of cumulated heat at certain age

# define target time
target_h = 1.5

# get cumlated heat flows for each sample
cum_h = tam.get_cumulated_heat_at_hours(target_h=target_h, cutoff_min=10)
print(cum_h)

# show cumulated heat plot
ax = tam.plot(t_unit="h", y="normalized_heat_j_g", y_unit_milli=False)

# guide to the eye line
ax.axvline(target_h, color="gray", alpha=0.5, linestyle=":")

# set upper limits
ax.set_ylim(top=250)
ax.set_xlim(right=12)
# show plot
tacalorimetry.plt.show()

# %% get peaks

peaks = tam.get_peaks(show_plot=True)


# %% get onsets

# get onsets
onsets = tam.get_peak_onsets(
    gradient_threshold=0.000004, rolling=10, exclude_discarded_time=True, show_plot=True
)

# %% show detected onsets per samppl

import seaborn as sns

sns.barplot(
    data=onsets,
    x="time_s",
    y=[os.path.basename(i) for i in onsets["sample"]],
    color="#451256",
)
