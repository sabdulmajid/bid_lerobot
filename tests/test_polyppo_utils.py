import torch

from lerobot.common.polyppo.utils import select_device


def test_select_device_applies_gpu_id_to_unindexed_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = select_device("cuda", gpu_id=1)

    assert str(device) == "cuda:1"


def test_select_device_keeps_explicit_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = select_device("cuda:0", gpu_id=1)

    assert str(device) == "cuda:0"
