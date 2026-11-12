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
    "__version__",
    "MittagLefflerPade34",
    "mittag_leffler_pade34",
    "AbelCausalAttention",
    "AbelTransformerLayer",
    "LogExpMultKAN",
    "GaussianBellGate",
    "MittagLefflerSpectralHead",
    "TemperatureTopKMoE",
]
