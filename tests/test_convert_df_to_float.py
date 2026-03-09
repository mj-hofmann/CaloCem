import numpy as np
import pandas as pd

from calocem.utils import convert_df_to_float


def test_convert_df_to_float_converts_empty_strings_to_nan_in_numeric_columns():
    df = pd.DataFrame(
        {
            "numeric_with_empty": ["1.5", "", "  ", "3"],
            "text": ["alpha", "beta", "", "gamma"],
            "only_empty": ["", " ", "\t", ""],
        }
    )

    out = convert_df_to_float(df.copy())

    assert out["numeric_with_empty"].dtype == float
    assert out["numeric_with_empty"].tolist()[0] == 1.5
    assert np.isnan(out["numeric_with_empty"].tolist()[1])
    assert np.isnan(out["numeric_with_empty"].tolist()[2])
    assert out["numeric_with_empty"].tolist()[3] == 3.0

    assert out["text"].tolist() == ["alpha", "beta", "", "gamma"]

    assert out["only_empty"].dtype == float
    assert out["only_empty"].isna().all()
