
import pathlib

import pytest

from calocem import tacalorimetry

def test_read_1st_gen():
    # path
    path = pathlib.Path(__file__).parent.parent / "calocem" / "DATA"
    
    files = ["gen1_calofile2.csv", "gen1_calofile3.csv" ]
    files_regex = "|".join(files)
    # init object
    tam = tacalorimetry.Measurement(path, auto_clean=False, regex=files_regex, show_info=True, cold_start=True)
    
    assert set(files) == set(tam._data.sample_short.unique() + ".csv")