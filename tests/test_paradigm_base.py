# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for paradigm base classes
#
# Tests for: metabci.brainda.paradigms.base and the concrete paradigm gating.

import pytest

from metabci.brainda.paradigms import (
    P300,
    SSVEP,
    BaseParadigm,
    MotorImagery,
    MovementIntention,
    aVEP,
)


class _FakeDataset:
    """Duck-typed dataset carrying only the metadata paradigms inspect."""

    def __init__(self, paradigm="imagery", events=None):
        self.paradigm = paradigm
        self.events = (
            events
            if events is not None
            else {
                "left_hand": (1, (0.0, 1.0)),
                "right_hand": (2, (0.5, 1.5)),
            }
        )
        self.channels = ["CZ", "OZ"]
        self.srate = 250


class _MinimalParadigm(BaseParadigm):
    """Minimal concrete paradigm exposing the base-class machinery."""

    def is_valid(self, dataset):
        return True

    def _get_single_subject_data(self, dataset, subject_id, verbose=False):
        return None


PARADIGM_CASES = [
    (aVEP, "aVEP"),
    (MotorImagery, "imagery"),
    (MovementIntention, "movement_intention"),
    (P300, "p300"),
    (SSVEP, "ssvep"),
]


@pytest.mark.parametrize(
    "para_cls, tag",
    PARADIGM_CASES,
    ids=[cls.__name__ for cls, _ in PARADIGM_CASES],
)
class TestParadigmGating:
    """is_valid accepts its own paradigm tag and rejects the others."""

    def test_is_valid_positive(self, para_cls, tag):
        assert para_cls().is_valid(_FakeDataset(paradigm=tag)) is True

    def test_is_valid_negative(self, para_cls, tag):
        other_tag = next(t for _, t in PARADIGM_CASES if t != tag)
        assert para_cls().is_valid(_FakeDataset(paradigm=other_tag)) is False


class TestSelectChannels:
    """BaseParadigm normalizes selected channel names to uppercase."""

    def test_channels_uppercase(self):
        p = _MinimalParadigm(channels=["cz", "oz"])
        assert p.select_channels == ["CZ", "OZ"]

    def test_channels_none_kept(self):
        p = _MinimalParadigm()
        assert p.select_channels is None


class TestMapEventsIntervals:
    """BaseParadigm._map_events_intervals selection modes."""

    def _fake(self):
        return _FakeDataset(
            events={
                "left_hand": (1, (0.0, 1.0)),
                "right_hand": (2, (0.5, 1.5)),
            }
        )

    def test_all_events_default_intervals(self):
        p = _MinimalParadigm()
        events, intervals = p._map_events_intervals(self._fake())
        assert events == {"left_hand": 1, "right_hand": 2}
        assert intervals == {"left_hand": (0.0, 1.0), "right_hand": (0.5, 1.5)}

    def test_event_subset(self):
        p = _MinimalParadigm(events=["left_hand"])
        events, intervals = p._map_events_intervals(self._fake())
        assert events == {"left_hand": 1}
        assert intervals == {"left_hand": (0.0, 1.0)}

    def test_single_shared_interval(self):
        p = _MinimalParadigm(intervals=[(-0.5, 1.0)])
        _, intervals = p._map_events_intervals(self._fake())
        assert intervals == {"left_hand": (-0.5, 1.0), "right_hand": (-0.5, 1.0)}

    def test_per_event_intervals(self):
        p = _MinimalParadigm(
            events=["left_hand", "right_hand"],
            intervals=[(0.0, 1.0), (0.5, 2.0)],
        )
        _, intervals = p._map_events_intervals(self._fake())
        assert intervals == {"left_hand": (0.0, 1.0), "right_hand": (0.5, 2.0)}

    def test_interval_count_mismatch_raises(self):
        p = _MinimalParadigm(
            events=["left_hand", "right_hand"],
            intervals=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        )
        with pytest.raises(ValueError):
            p._map_events_intervals(self._fake())


class TestParadigmStr:
    def test_str_returns_class_name(self):
        assert str(P300()) == "P300"
