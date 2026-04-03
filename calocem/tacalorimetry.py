"""
Compatibility shim — this module is deprecated.

Use the following imports instead:

    from calocem.measurement import Measurement
    from calocem.processparams import ProcessingParameters

This module will be removed in a future major version.
"""
import warnings

warnings.warn(
    "calocem.tacalorimetry is deprecated and will be removed in a future version. "
    "Use 'from calocem.measurement import Measurement' instead.",
    DeprecationWarning,
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
