import numpy as np
import numpy.typing as npt

def compute_bitwise_hamming_distance(x: npt.NDArray[np.uint8], y: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint32]: ...
def compute_bitwise_hamming_distances(
    x: npt.NDArray[np.uint8], ys: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint32]: ...
