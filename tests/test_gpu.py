import numpy as np
import pytest
import torch

from muzero_research.config import preset
from muzero_research.experiment import Experiment


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA hardware is unavailable")
@pytest.mark.parametrize("method", ["mcts", "beam", "genetic", "nes"])
def test_gpu_training_and_cpu_checkpoint_loading(method, tmp_path):
    config = preset("lunarlander", method, smoke=True)
    config.training.steps = 1
    config.runtime.device = "cuda"
    config.output_dir = str(tmp_path / method)
    experiment = Experiment(config)
    assert np.isfinite(experiment.train(tensorboard=False)["total_loss"])
    config.runtime.device = "cpu"
    restored = Experiment.from_checkpoint(experiment.save(), config=config)
    assert np.isfinite(restored.evaluate()[0]["raw_return"])
