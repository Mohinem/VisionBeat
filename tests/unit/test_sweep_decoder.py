from __future__ import annotations

import pytest

from visionbeat.sweep_decoder import SweepConfig, build_sweep_configs, format_sweep_slug


def test_build_sweep_configs_builds_cartesian_grid_in_stable_order() -> None:
    configs = build_sweep_configs(
        thresholds=[0.55, 0.60],
        cooldowns=[6, 8],
        max_gaps=[1, 2],
    )

    assert configs == (
        SweepConfig(threshold=0.55, cooldown_frames=6, max_gap_frames=1),
        SweepConfig(threshold=0.55, cooldown_frames=6, max_gap_frames=2),
        SweepConfig(threshold=0.55, cooldown_frames=8, max_gap_frames=1),
        SweepConfig(threshold=0.55, cooldown_frames=8, max_gap_frames=2),
        SweepConfig(threshold=0.6, cooldown_frames=6, max_gap_frames=1),
        SweepConfig(threshold=0.6, cooldown_frames=6, max_gap_frames=2),
        SweepConfig(threshold=0.6, cooldown_frames=8, max_gap_frames=1),
        SweepConfig(threshold=0.6, cooldown_frames=8, max_gap_frames=2),
    )


def test_format_sweep_slug_uses_fixed_threshold_precision() -> None:
    config = SweepConfig(threshold=0.6, cooldown_frames=10, max_gap_frames=2)

    assert format_sweep_slug(config) == "th_0p60_cd_10_gap_2"


@pytest.mark.parametrize(
    ("thresholds", "cooldowns", "max_gaps", "message"),
    [
        ([], [6], [1], "thresholds"),
        ([0.55], [], [1], "cooldowns"),
        ([0.55], [6], [], "max_gaps"),
        ([1.2], [6], [1], "threshold"),
        ([0.55], [-1], [1], "cooldown_frames"),
        ([0.55], [6], [-1], "max_gap_frames"),
    ],
)
def test_build_sweep_configs_rejects_invalid_inputs(
    thresholds: list[float],
    cooldowns: list[int],
    max_gaps: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_sweep_configs(
            thresholds=thresholds,
            cooldowns=cooldowns,
            max_gaps=max_gaps,
        )
