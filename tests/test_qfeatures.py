import datetime as dt

import numpy as np

from staysearch import qfeatures as qf


def test_lead_buckets_match_contract():
    lead = np.array([0, 1, 2, 7, 8, 14, 15, 30, 31, 60, 61, 400])
    assert qf.lead_bucket(lead).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_dow_monday_first():
    d = dt.date(2026, 10, 5)  # a Monday
    assert qf.dow(np.array([d.toordinal(), d.toordinal() + 6])).tolist() == [0, 6]


def test_query_base_one_hots():
    x = qf.query_base(np.array([10.0]), np.array([True]), np.array([2]), np.array([3]), np.array([20]),
                      np.array([dt.date(2026, 10, 7).toordinal()]))
    assert x.shape == (1, 17)
    assert x[0, 4:10].tolist() == [0, 0, 0, 1, 0, 0]
    assert x[0, 10:17].tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert abs(x[0, 2] - 2 / 16) < 1e-7


def test_history_block_and_cross_features():
    emb = np.eye(4, 64, dtype=np.float32)
    price = np.array([100.0, 200.0, np.nan, 50.0])
    room = np.array([0, 1, 0, 0])
    h = qf.history_block([np.array([0, 1]), np.array([], dtype=np.int64)], emb, price, room)
    assert h[0, 0] == 1 and h[1].sum() == 0
    assert np.allclose(h[0, 2:6], [0.5, 0.5, 0, 0])
    x = qf.cross_features(np.array([1.0, 2.0]), np.array([47.6, 47.61]), np.array([-122.3, -122.3]),
                          np.array([100.0, np.nan]), np.array([1, 2]), np.array([4, 2]), np.array([0, 1]),
                          emb[:2], 47.6, -122.3, 25.0, 2, 3, 10, h[0])
    assert x.shape == (2, qf.N_CROSS)
    assert x[1, 2] == 0.0 and x[0, 7] == 1.0
    assert abs(x[0, 11] - 0.5) < 1e-7
