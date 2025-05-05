import numpy as np
from typing import Literal
from scipy.spatial.transform import Rotation as R


def matrix2vector(
    matrix: np.ndarray,
    rotation_type: Literal['euler', 'quat'] = 'euler'
) -> np.ndarray:
    assert rotation_type in ['euler', 'quat']
    translation = matrix[..., :3, 3]
    rotation = matrix[..., :3, :3]
    rotation = R.from_matrix(rotation)
    if rotation_type == 'euler':
        rotation = rotation.as_euler('xyz')
    elif rotation_type == 'quat':
        rotation = rotation.as_quat()
    vector = np.concatenate([translation, rotation], axis=-1)
    return vector


def vector2matrix(
    vector: np.ndarray,
    rotation_type: Literal['euler', 'quat'] = 'euler'
) -> np.ndarray:
    assert rotation_type in ['euler', 'quat']
    translation = vector[..., :3]
    rotation = vector[..., 3:]
    if rotation_type == 'euler':
        assert rotation.shape[-1] == 3
        rotation = R.from_euler('xyz', rotation).as_matrix()
    elif rotation_type == 'quat':
        assert rotation.shape[-1] == 4
        rotation = R.from_quat(rotation).as_matrix()
    matrix = np.zeros((*translation.shape[:-1], 4, 4))
    matrix[..., :3, 3] = translation
    matrix[..., :3, :3] = rotation
    matrix[..., 3, 3] = 1
    return matrix