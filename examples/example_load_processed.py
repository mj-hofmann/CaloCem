from pathlib import Path
from calocem.tacalorimetry import Measurement



datapath = Path(__file__).parent.parent / "calocem" / "DATA"
#datapath = parentfolder / "tmp"

# experiments via class
tam = Measurement(
    folder=datapath,
    regex=r"processed.*",
    show_info=True,
    auto_clean=False,
    cold_start=True,
    new_code=True,
    processed=True,
)

print(tam._data.head(10))
