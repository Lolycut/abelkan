"""
abelkan: Pole-free Pade Mittag-Leffler spectral layers, Abel power-law attention,
and neuro-symbolic KAN primitives for PyTorch.
"""

from .attention import AbelCausalAttention, AbelTransformerLayer
from .kan import GaussianBellGate, LogExpMultKAN, MittagLefflerSpectralHead
from .moe import TemperatureTopKMoE
from .pade import MittagLefflerPade34, mittag_leffler_pade34

__version__ = "0.1.0"

__all__ = [
    "AbelCausalAttention",
    "AbelTransformerLayer",
    "GaussianBellGate",
    "LogExpMultKAN",
    "MittagLefflerPade34",
    "MittagLefflerSpectralHead",
    "TemperatureTopKMoE",
    "__version__",
    "mittag_leffler_pade34",
]
