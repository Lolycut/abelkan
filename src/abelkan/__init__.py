"""
abelkan: Pole-free Pade Mittag-Leffler spectral layers, Abel power-law attention,
and neuro-symbolic KAN primitives for PyTorch.
"""

from .attention import AbelCausalAttention, AbelTransformerLayer
from .kan import (
    AdaptiveMoEMittagLefflerHead,
    AnchorResidualMittagLefflerHead,
    ComplexMittagLefflerSpectralHead,
    GaussianBellGate,
    LogExpMultKAN,
    MittagLefflerSpectralHead,
)
from .moe import TemperatureTopKMoE
from .pade import MittagLefflerPade34, mittag_leffler_pade34

__version__ = "0.6.0"

__all__ = [
    "AbelCausalAttention",
    "AbelTransformerLayer",
    "AdaptiveMoEMittagLefflerHead",
    "AnchorResidualMittagLefflerHead",
    "ComplexMittagLefflerSpectralHead",
    "GaussianBellGate",
    "LogExpMultKAN",
    "MittagLefflerPade34",
    "MittagLefflerSpectralHead",
    "TemperatureTopKMoE",
    "__version__",
    "mittag_leffler_pade34",
]

from .soe import soe_sequential_scan, soe_associative_scan, soe_chunkwise_scan, to_bfloat16_sim
from .triton_soe import soe_chunkwise_forward
