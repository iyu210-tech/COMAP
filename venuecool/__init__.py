"""VenueCool: transient section-resolved cooling assessment for open-air and semi-open venues.

The scientific engine (this package) is independent of the Streamlit dashboard in ``app/``.
Internal units are SI: K differences / degC temperatures, Pa, kg, s, W (power), J (energy).
Reported energies are converted to kWh only at the reporting boundary.
"""

import os
from pathlib import Path

# pythermalcomfort uses numba with on-disk caching; make sure a writable cache directory exists.
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".numba_cache")
)

__version__ = "0.1.0"
