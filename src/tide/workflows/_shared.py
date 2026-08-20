import os
from contextlib import contextmanager
from typing import Iterator, Sequence

import numpy as np

from tide.core import physics, tractography

SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class WorkflowError(RuntimeError):
    pass


@contextmanager
def single_thread_child_environment() -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in SINGLE_THREAD_ENV}
    os.environ.update(SINGLE_THREAD_ENV)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def configure_worker_environment() -> None:
    os.environ.update(SINGLE_THREAD_ENV)


def split_vectors_by_streamline(
    vectors: np.ndarray,
    streamlines: Sequence[np.ndarray],
) -> list[np.ndarray]:
    result = []
    offset = 0
    for streamline in streamlines:
        point_count = len(streamline)
        result.append(vectors[offset : offset + point_count])
        offset += point_count
    return result


def calculate_target_in_field_metric(
    streamlines: Sequence[np.ndarray],
    e_field_vectors: Sequence[np.ndarray],
    *,
    roi_center: Sequence[float],
    roi_size_mm: float,
    activation_length_mm: float,
    max_angular_deviation_deg: float,
) -> float:
    streamlines_for_validation = list(streamlines)
    vectors_for_validation = list(e_field_vectors)
    if max_angular_deviation_deg > 0:
        (
            streamlines_for_validation,
            vectors_for_validation,
            _,
        ) = tractography.filter_by_angular_deviation(
            streamlines_for_validation,
            e_field_vectors=vectors_for_validation,
            max_angle_deg=max_angular_deviation_deg,
            roi_center=roi_center,
            roi_radius=roi_size_mm,
        )

    midpoint_streamlines, af_values, segment_lengths = physics.calculate_scalar_map(
        streamlines_for_validation,
        vectors_for_validation,
        mode="af",
    )
    roi_masks, _ = tractography.get_roi_masks(
        midpoint_streamlines,
        roi_size_mm,
        roi_center,
    )

    scores = []
    for af_value, lengths, mask in zip(af_values, segment_lengths, roi_masks):
        af_in_roi = af_value[mask]
        if not np.all(np.isfinite(af_in_roi)):
            continue
        scores.append(
            physics.get_max_contiguous_threshold(
                np.abs(af_in_roi),
                lengths[mask],
                activation_length_mm,
            )
        )

    return physics.median_of_top_percentile(np.array(scores), 95.0)
