from .rae_da3 import RAE_DA3

try:
    from .rae_vggt import RAE_VGGT
except (ImportError, AttributeError):
    RAE_VGGT = None
