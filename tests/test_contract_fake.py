from contract import check_contract

from kr.fake_model import FakeModel


def test_fake_meets_contract():
    check_contract(FakeModel({"The capital of France is": " Paris"}))
