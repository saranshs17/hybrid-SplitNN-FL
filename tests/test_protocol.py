import pytest
from pathlib import Path
import yaml
from src.sim import run_simulation as rs

def test_run_round_smoke(tmp_path):
    cfg = {
        "num_clients": 1,
        "num_fog": 1,
        "batch_size": 4,
        "dataset": "random",
        "epochs": 1,
        "device": "cpu"
    }
    rs.run(cfg)
    assert True
