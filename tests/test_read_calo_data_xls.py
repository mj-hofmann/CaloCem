import pathlib

import pysnooper
import pytest

from calocem.measurement import Measurement


#
# run "time check" on each sample file
# see https://www.youtube.com/watch?v=DhUpxWjOhME (~ at 12:20)
# to use the the test:
#       1) install the module locally with "pip install -e ."
#       2) run "pytest" from shell
#
@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("excel_example4.xls", 238061),
        ("excel_example2.xls", 499440),
        ("excel_example3.xls", 336000),
    ],
)
#@pysnooper.snoop()
def test_last_time_entry(test_input, expected):

    # path
    path = pathlib.Path().cwd() / "calocem" / "DATA"

    # get data
    data = Measurement(path, regex=test_input)
    data = data.get_data()
    # check for None return
    if data is None:
        # check
        assert data == expected
    else:
        # get "last time for a file
        last_time = int(data.tail(1)["time_s"].values[0])
        # actual test
        assert last_time == int(expected)
