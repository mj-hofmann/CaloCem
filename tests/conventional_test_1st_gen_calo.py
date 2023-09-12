import matplotlib.pyplot as plt

import sys
from pathlib import Path

parentfolder = Path(__file__).cwd().parent
sys.path.insert(0, parentfolder.as_posix())

import TAInstCalorimetry.tacalorimetry as ta

datapath = parentfolder / "TAInstCalorimetry" / "DATA"
#datapath = parentfolder / "tmp"

# experiments via class
tam = ta.Measurement(
    folder=datapath,
    regex=r".*1st.*.csv",
    show_info=True,
    auto_clean=False,
    cold_start=True,
)

fig, ax = plt.subplots()
tam.plot()
ax.set_xlim(0, 5)
ta.plt.show()
