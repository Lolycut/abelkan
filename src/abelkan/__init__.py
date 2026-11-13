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
from .soe import (
    soe_associative_scan,
    soe_chunkwise_scan,
    soe_sequential_scan,
    to_bfloat16_sim,
)
from .triton_soe import soe_chunkwise_forward

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
    "mittag_leffler_pade34",
    "soe_associative_scan",
    "soe_chunkwise_forward",
    "soe_chunkwise_scan",
    "soe_sequential_scan",
    "to_bfloat16_sim",
]
