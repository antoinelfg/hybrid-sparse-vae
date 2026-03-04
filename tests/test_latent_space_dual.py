import torch
import pytest
from modules.latent_space import StructuredLatentSpace

@pytest.fixture
def latent_space_temp():
    return StructuredLatentSpace(
        input_dim=256,
        n_atoms=64,
        latent_dim=128,
        temporal_mode=True,
    )

@pytest.fixture
def latent_space_dense():
    return StructuredLatentSpace(
        input_dim=256,
        n_atoms=64,
        latent_dim=128,
        temporal_mode=False,
    )

def test_temporal_shapes(latent_space_temp):
    B = 4
    C = 256
    T = 16
    h = torch.randn(B, C, T)
    z, info = latent_space_temp(h)
    
    assert z.shape == (B, 64, T)
    assert info["B"].shape == (B, 64, T)
    assert info["gamma"].shape == (B, 64, T)
    assert info["delta"].shape == (B, 64, T)

def test_dense_shapes(latent_space_dense):
    B = 4
    C = 256
    h = torch.randn(B, C)
    z, info = latent_space_dense(h)
    
    assert z.shape == (B, 128)
    assert info["B"].shape == (B, 64)
    assert info["gamma"].shape == (B, 64)
    assert info["delta"].shape == (B, 64)

def test_both():
    pytest.main(["-v", "tests/test_latent_space_dual.py"])

if __name__ == "__main__":
    test_both()
