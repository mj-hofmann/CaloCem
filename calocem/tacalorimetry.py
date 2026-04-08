"""
This module has been removed as of version 0.3.0.

The legacy ``calocem.tacalorimetry.Measurement`` implementation has been
replaced by the refactored ``calocem.measurement.Measurement``. The two
implementations are not guaranteed to produce identical numerical results.

Migrate your imports:

    # old
    from calocem.tacalorimetry import Measurement

    # new
    from calocem import Measurement
    # or
    from calocem.measurement import Measurement

Users who require the old behaviour should pin to ``calocem<0.3.0``.
"""
import warnings

warnings.warn(
    "calocem.tacalorimetry has been removed in version 0.3.0. "
    "The old implementation is no longer available. "
    "Migrate to 'from calocem import Measurement' or pin to calocem<0.3.0 "
    "if you require the previous behaviour.",
    FutureWarning,
    stacklevel=2,
)

import matplotlib.pyplot as plt  # noqa: E402 — kept for tacalorimetry.plt back-compat

from .exceptions import (  # noqa: F401
    AddMetaDataSourceException,
    AutoCleanException,
    ColdStartException,
    DataProcessingException,
    FileReadingException,
)
from .measurement import Measurement  # noqa: F401
from .processparams import ProcessingParameters  # noqa: F401
