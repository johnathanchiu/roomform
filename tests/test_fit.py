"""Fit stage: box-frame fit correctness and the degenerate-mesh gates."""

import math

import numpy as np
import pytest
import trimesh

from roomform.pipe.objects.reconstruction.fit import fit_to_box, qc


def scene_of(mesh: trimesh.Trimesh) -> trimesh.Scene:
    return trimesh.Scene(mesh)


def test_fit_maps_mesh_onto_box() -> None:
    mesh = scene_of(trimesh.creation.box(extents=[2.0, 1.0, 1.0]))
    center = np.array([5.0, 3.0, 0.4])
    size = np.array([1.0, 0.5, 0.8])
    fitted = fit_to_box(mesh, center, size, heading=0.0)
    lo, hi = fitted.bounds
    assert np.allclose(hi - lo, size, atol=1e-6)
    # floor-standing box: mesh bottom pinned to box bottom
    assert math.isclose(lo[2], -size[2] / 2, abs_tol=1e-6)


def test_fit_respects_heading() -> None:
    # a scene-aligned mesh whose long axis already matches the rotated
    # box keeps that orientation and lands exactly on the box extents
    mesh = scene_of(trimesh.creation.box(extents=[1.0, 2.0, 1.0]))
    size = np.array([2.0, 1.0, 1.0])  # box-local: long axis is x
    fitted = fit_to_box(mesh, np.zeros(3), size, heading=math.pi / 2)
    lo, hi = fitted.bounds
    # heading pi/2 puts the box's long axis along world y
    assert (hi - lo)[1] == pytest.approx(2.0, abs=1e-6)
    assert (hi - lo)[0] == pytest.approx(1.0, abs=1e-6)


def test_misoriented_mesh_rejected() -> None:
    # same mesh 90 degrees off its box: the distortion gate refuses to
    # silently squash it into place
    mesh = scene_of(trimesh.creation.box(extents=[2.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="distortion"):
        fit_to_box(
            mesh, np.zeros(3), np.array([2.0, 1.0, 1.0]), heading=math.pi / 2
        )


def test_flat_card_rejected_by_qc() -> None:
    card = scene_of(trimesh.creation.box(extents=[2.0, 1.5, 0.02]))
    assert (
        qc(card, box_size=np.array([2.0, 1.5, 1.2]), heading=0.0) is not None
    )


def test_card_rejected_by_distortion_gate() -> None:
    # passes thinness relative to a thin box, but the fit would need >3x
    # anisotropic stretch — the distortion gate catches it
    card = scene_of(trimesh.creation.box(extents=[2.0, 2.0, 0.2]))
    with pytest.raises(ValueError, match="distortion"):
        fit_to_box(card, np.zeros(3), np.array([2.0, 1.0, 1.0]), heading=0.0)


def test_solid_mesh_passes_qc() -> None:
    box = scene_of(trimesh.creation.box(extents=[1.8, 0.9, 0.7]))
    assert qc(box, box_size=np.array([2.0, 1.0, 0.8]), heading=0.0) is None
