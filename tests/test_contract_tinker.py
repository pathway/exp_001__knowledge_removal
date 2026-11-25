"""Live contract test against Tinker. Costs well under $0.001. Run with TINKER_LIVE=1."""
import os

import pytest
from contract import check_contract

pytestmark = pytest.mark.skipif(os.environ.get("TINKER_LIVE") != "1", reason="set TINKER_LIVE=1 to spend money")


def test_tinker_meets_contract():
    from kr.config import BASE_MODEL
    from kr.tinker_model import TinkerModel
    check_contract(TinkerModel.open(BASE_MODEL, rank=1))
