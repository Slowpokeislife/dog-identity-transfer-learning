"""Lightweight smoke tests (no pretrained weight download)."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dog_identity.train import run_epoch


def test_metrics_perfect():
    class Constant(nn.Module):
        def forward(self, images):
            return images[:, 0:1]
    dataset = TensorDataset(torch.tensor([[-4.], [4.], [-2.], [2.]]),
                            torch.tensor([0, 1, 0, 1]))
    result = run_epoch(Constant(), DataLoader(dataset, batch_size=3),
                       nn.BCEWithLogitsLoss(), torch.device("cpu"))
    assert result["accuracy"] == 1.0
    assert result["precision_bo"] == 1.0
    assert result["recall_bo"] == 1.0
    assert result["f1_bo"] == 1.0
