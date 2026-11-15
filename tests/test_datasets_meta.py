# -*- coding: utf-8 -*-
# [METABCI-PYTEST-SUITE] Tests for dataset metadata (offline-only)
#
# Tests for: metabci.brainda.datasets (base classes and concrete loaders)
#
# No data is downloaded — every concrete dataset class is instantiated and
# only its metadata attributes are inspected.

import pytest

from metabci.brainda.datasets import (
    BETA,
    BNCI2014001,
    BNCI2014004,
    CBCIC2019001,
    CBCIC2019004,
    AlexMI,
    Cattan_P300,
    Cho2017,
    MunichMI,
    Nakanishi2015,
    PhysionetME,
    PhysionetMI,
    Schirrmeister2017,
    Wang2016,
    Weibo2014,
    Xu2018MinaVep,
    Zhou2016,
)
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainda.paradigms import (
    P300,
    SSVEP,
    MotorImagery,
    MovementIntention,
    aVEP,
)

# `matchingpennies` is excluded: its __init__ calls mne_data_path() and
# therefore starts downloading the BIDS archive, which breaks offline
# construction. Deferring that download to data_path() would make it
# testable here.
DATASET_CLASSES = [
    AlexMI,
    BETA,
    BNCI2014001,
    BNCI2014004,
    Cattan_P300,
    CBCIC2019001,
    CBCIC2019004,
    Cho2017,
    MunichMI,
    Nakanishi2015,
    PhysionetME,
    PhysionetMI,
    Schirrmeister2017,
    Wang2016,
    Weibo2014,
    Xu2018MinaVep,
    Zhou2016,
]

PARADIGM_INSTANCES = [
    aVEP(),
    MotorImagery(),
    MovementIntention(),
    P300(),
    SSVEP(),
]


@pytest.mark.parametrize(
    "ds_cls", DATASET_CLASSES, ids=[cls.__name__ for cls in DATASET_CLASSES]
)
class TestDatasetMetadata:
    """Offline metadata health checks for every concrete dataset."""

    def test_constructs_offline(self, ds_cls):
        """Every concrete dataset can be instantiated without any download."""
        ds = ds_cls()
        assert isinstance(ds, BaseDataset)

    def test_basic_metadata(self, ds_cls):
        """dataset_code / subjects / srate / paradigm are sane."""
        ds = ds_cls()
        assert isinstance(ds.dataset_code, str) and len(ds.dataset_code) > 0
        assert len(ds.subjects) > 0
        assert all(isinstance(s, (int, str)) for s in ds.subjects)
        assert isinstance(ds.srate, (int, float)) and ds.srate > 0
        assert isinstance(ds.paradigm, str) and len(ds.paradigm) > 0

    def test_channels_normalized_uppercase(self, ds_cls):
        """BaseDataset normalizes channel names to uppercase."""
        ds = ds_cls()
        assert len(ds.channels) > 0
        assert all(ch == ch.upper() for ch in ds.channels)

    def test_events_well_formed(self, ds_cls):
        """events maps str names to (event_id, (tmin, tmax)) with tmin < tmax."""
        ds = ds_cls()
        assert len(ds.events) > 0
        for name, (event_id, interval) in ds.events.items():
            assert isinstance(name, str)
            assert isinstance(event_id, (int, str))
            tmin, tmax = interval
            assert tmin < tmax

    def test_accepted_by_at_least_one_paradigm(self, ds_cls):
        """Every dataset is consumable by at least one registered paradigm."""
        ds = ds_cls()
        assert any(paradigm.is_valid(ds) for paradigm in PARADIGM_INSTANCES)

    def test_str_repr(self, ds_cls):
        """str/repr render the dataset code."""
        ds = ds_cls()
        assert ds.dataset_code in str(ds)
        assert str(ds) == repr(ds)
