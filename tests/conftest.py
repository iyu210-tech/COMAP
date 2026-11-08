import copy
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".numba_cache"))

MINI = ROOT / "tests" / "data" / "mini.yaml"


def q(v, unit, src="test"):
    return {"value": v, "unit": unit, "source": src}


@pytest.fixture(scope="session")
def mini_raw():
    from venuecool.config import expand_templates, load_raw
    return expand_templates(load_raw(MINI))


@pytest.fixture(scope="session")
def mini_study():
    from venuecool.runner import Study
    return Study(MINI)


@pytest.fixture(scope="session")
def mini_base(mini_study):
    return mini_study.run_scenario("baseline")


def study_with(raw, mutate):
    from venuecool.runner import Study
    r = copy.deepcopy(raw)
    mutate(r)
    return Study(r)


def sealed(r, zero_people=True):
    """Remove outdoor exchange (and optionally occupants) from every section."""
    for s in r["sections"]:
        s["outdoor"]["wind_coefficient"] = q(0.0, "-")
        s["outdoor"]["min_flow"] = q(0.0, "m3/s")
        if zero_people:
            s["occupancy"]["column"] = "zero"
