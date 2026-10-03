import numpy as np

from staysearch.samplers import Groups, Samplers, guard_mask, pair_key


def make(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    city = rng.integers(0, 3, n)
    nbhd = city * 10 + rng.integers(0, 10, n)
    s2 = nbhd * 5 + rng.integers(0, 5, n)
    acc = rng.integers(1, 9, n)
    price = rng.lognormal(5, 0.5, n)
    price[rng.random(n) < 0.05] = np.nan
    return Samplers(city, nbhd, s2, acc, price), city, nbhd, s2, acc, price


def test_groups_draw_stays_in_bucket():
    keys = np.array([5, 5, 7, 7, 7, 9])
    g = Groups(keys)
    rows, ok = g.draw(np.array([7, 9, 4]), 50, np.random.default_rng(0))
    assert set(keys[rows[0]]) == {7}
    assert list(ok) == [True, False, False]


def test_random_is_same_city():
    s, city, *_ = make()
    pos = np.arange(500)
    neg = s.random(pos, 4, np.random.default_rng(1))
    assert (city[neg] == city[pos][:, None]).all()


def test_geo_is_same_cell_or_nbhd_and_band():
    s, city, nbhd, s2, acc, _ = make()
    pos = np.arange(500)
    neg = s.geo(pos, 4, np.random.default_rng(2))
    band = np.searchsorted([2, 4, 6], acc, side="left")
    same_cell = (s2[neg] == s2[pos][:, None]) & (band[neg] == band[pos][:, None])
    same_nbhd = (nbhd[neg] == nbhd[pos][:, None]) & (band[neg] == band[pos][:, None])
    assert (same_cell | same_nbhd | (city[neg] == city[pos][:, None])).all()
    assert same_cell.mean() > 0.5


def test_price_within_20_percent_when_possible():
    s, city, nbhd, s2, acc, price = make(5000)
    pos = np.nonzero(~np.isnan(price))[0][:500]
    neg = s.price(pos, 4, np.random.default_rng(3))
    ratio = price[neg] / price[pos][:, None]
    in_band = (nbhd[neg] == nbhd[pos][:, None]) & (ratio > 0.8 - 1e-9) & (ratio < 1.2 + 1e-9)
    assert in_band.mean() > 0.9


def test_guard_masks_positive_and_reviewed():
    reviewed = np.sort(pair_key(np.array([1, 1]), np.array([100, 200])))
    neg = np.array([[100, 300, 200, 50]])
    m = guard_mask(np.array([1]), neg, np.array([50]), reviewed)
    assert m.tolist() == [[False, True, False, False]]
