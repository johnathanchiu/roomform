import numpy as np

from roomform.export import _cell_points


def test_cell_points_use_centers_and_learned_offsets():
    cells = np.array([[1, 2, 3]])
    offsets = np.zeros((3, 3, 4, 5), dtype=np.float32)
    offsets[:, 1, 2, 3] = [0.25, -0.25, 0.5]

    points = _cell_points(cells, vox=0.08, offsets=offsets)

    np.testing.assert_allclose(points[0], [0.14, 0.18, 0.32])
