from dataclasses import dataclass
import numpy as np

@dataclass
class BinaryMask:
    name: str = None
    path: str = None
    img: np.ndarray = None