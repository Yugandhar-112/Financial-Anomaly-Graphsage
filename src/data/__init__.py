from .elliptic_loader import load_elliptic, EllipticStats
from .synthetic import get_synthetic_dataset


def load_dataset(synthetic: bool = False, data_dir: str = "data/raw/elliptic", **kwargs):
    """
    Single entry point used by train.py.

    synthetic=False (default): loads the real Elliptic dataset from `data_dir`.
    synthetic=True: builds the synthetic demo graph instead (no CSVs needed) --
        useful for CI, quick sanity checks, or offline demos. Never used for
        reported results.

    Returns (data, stats) where stats is None for the synthetic path.
    """
    if synthetic:
        return get_synthetic_dataset(**kwargs), None
    return load_elliptic(data_dir=data_dir, **kwargs)