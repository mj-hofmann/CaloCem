import matplotlib.pyplot as plt

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

# apply Tian-correction
tam.apply_tian_correction(
    tau=300, 
    smoothing=1.5e-2
    )

# loop samples
for sample, data in tam.iter_samples():
    print(sample)
    # fig, ax = plt.subplots()
    p = plt.plot(
        data["time_s"], data["normalized_heat_flow_w_g"], alpha=0.5, linestyle=":"
    )
    plt.plot(
        data["time_s"], data["normalized_heat_flow_w_g_tian"], color=p[0].get_color()
    )

# set limit
plt.xlim(0, 1000)
plt.ylim(0, 1.25)
plt.ylabel("normalized_heat_flow")
# show plot
plt.show()

# loop samples
for sample, data in tam.iter_samples():
    print(sample)
    # fig, ax = plt.subplots()
    p = plt.plot(
        data["time_s"], data["normalized_heat_j_g"], alpha=0.5, linestyle=":"
    )
    plt.plot(
        data["time_s"], data["normalized_heat_j_g_tian"], color=p[0].get_color()
    )

# set limit
plt.xlim(0, 1000)
plt.ylim(0, 150)
plt.ylabel("normalized_heat")
# show plot
plt.show()

# undo Tian-correction
tam.undo_tian_correction()

# plot
tam.plot(t_unit="s", y_unit_milli=False)
# set limit
plt.xlim(0, 1000)
plt.ylim(0, 1.25)
# show plot
plt.show()
