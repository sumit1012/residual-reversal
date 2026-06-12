"""Tests for residrev.factors — all mocked, no real HTTP or filesystem I/O."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from residrev.config import Config
from residrev.factors import (
    _sic_to_french12,
    get_factor_cov,
    get_ff_factors,
    get_sector_map,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path) -> Config:
    return Config(data_dir=str(tmp_path), start_date="2020-01-01", end_date="2020-12-31")


def _make_raw_ff5(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Fake FF5 DataFrame in raw percent form."""
    return pd.DataFrame(
        {
            "Mkt-RF": [0.5] * len(dates),
            "SMB": [0.1] * len(dates),
            "HML": [-0.2] * len(dates),
            "RMW": [0.3] * len(dates),
            "CMA": [0.0] * len(dates),
            "RF": [0.01] * len(dates),
        },
        index=dates,
    )


def _make_raw_umd(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"Mom   ": [0.4] * len(dates)}, index=dates)


# ---------------------------------------------------------------------------
# _sic_to_french12
# ---------------------------------------------------------------------------

class TestSicToFrench12:
    def test_consumer_nondurables(self):
        assert _sic_to_french12(2100) == "Consumer NonDurables"

    def test_energy(self):
        assert _sic_to_french12(1350) == "Energy"

    def test_finance(self):
        assert _sic_to_french12(6020) == "Finance"

    def test_healthcare(self):
        assert _sic_to_french12(8050) == "Healthcare"

    def test_utilities(self):
        assert _sic_to_french12(4910) == "Utilities"

    def test_telecom(self):
        assert _sic_to_french12(4813) == "Telecom"

    def test_business_equipment(self):
        assert _sic_to_french12(7372) == "Business Equipment"

    def test_other_for_unknown(self):
        assert _sic_to_french12(9999) == "Other"

    def test_other_for_zero(self):
        assert _sic_to_french12(0) == "Other"


# ---------------------------------------------------------------------------
# get_ff_factors
# ---------------------------------------------------------------------------

class TestGetFfFactors:
    def _mock_pdr(self, dates):
        ff5 = _make_raw_ff5(dates)
        umd = _make_raw_umd(dates)
        return {"ff5": ff5, "umd": umd}

    def test_loads_from_valid_cache(self, tmp_path):
        config = _make_config(tmp_path)
        dates = pd.date_range("2019-12-01", "2021-01-31", freq="B")
        cached_df = pd.DataFrame(
            {c: [0.001] * len(dates) for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]},
            index=dates,
        )
        cache_path = os.path.join(str(tmp_path), "factors_daily.parquet")
        cached_df.to_parquet(cache_path)

        with patch("residrev.factors.pdr.get_data_famafrench") as mock_pdr:
            result = get_ff_factors(config)
            mock_pdr.assert_not_called()

        assert list(result.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]

    def test_refetches_when_cache_out_of_range(self, tmp_path):
        config = _make_config(tmp_path)
        # Cache only covers 2019 — doesn't reach 2020 end
        dates = pd.date_range("2019-01-01", "2019-06-30", freq="B")
        cached_df = pd.DataFrame(
            {c: [0.001] * len(dates) for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]},
            index=dates,
        )
        cache_path = os.path.join(str(tmp_path), "factors_daily.parquet")
        cached_df.to_parquet(cache_path)

        fetch_dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
        ff5 = _make_raw_ff5(fetch_dates)
        umd = _make_raw_umd(fetch_dates)

        with patch("residrev.factors.pdr.get_data_famafrench") as mock_pdr, \
             patch("residrev.factors.mcal.get_calendar") as mock_cal:
            mock_pdr.side_effect = lambda name, **kw: {0: ff5} if "5_Factors" in name else {0: umd}
            cal_inst = MagicMock()
            cal_inst.valid_days.return_value = pd.DatetimeIndex(
                fetch_dates, dtype="datetime64[ns, UTC]"
            )
            mock_cal.return_value = cal_inst
            result = get_ff_factors(config)

        assert mock_pdr.call_count == 2

    def test_divides_by_100(self, tmp_path):
        config = _make_config(tmp_path)
        fetch_dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
        ff5 = _make_raw_ff5(fetch_dates)  # Mkt-RF raw = 0.5 percent
        umd = _make_raw_umd(fetch_dates)

        with patch("residrev.factors.pdr.get_data_famafrench") as mock_pdr, \
             patch("residrev.factors.mcal.get_calendar") as mock_cal:
            mock_pdr.side_effect = lambda name, **kw: {0: ff5} if "5_Factors" in name else {0: umd}
            cal_inst = MagicMock()
            cal_inst.valid_days.return_value = pd.DatetimeIndex(
                fetch_dates, dtype="datetime64[ns, UTC]"
            )
            mock_cal.return_value = cal_inst
            result = get_ff_factors(config)

        assert abs(result["Mkt-RF"].iloc[0] - 0.005) < 1e-10

    def test_has_seven_columns(self, tmp_path):
        config = _make_config(tmp_path)
        fetch_dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
        ff5 = _make_raw_ff5(fetch_dates)
        umd = _make_raw_umd(fetch_dates)

        with patch("residrev.factors.pdr.get_data_famafrench") as mock_pdr, \
             patch("residrev.factors.mcal.get_calendar") as mock_cal:
            mock_pdr.side_effect = lambda name, **kw: {0: ff5} if "5_Factors" in name else {0: umd}
            cal_inst = MagicMock()
            cal_inst.valid_days.return_value = pd.DatetimeIndex(
                fetch_dates, dtype="datetime64[ns, UTC]"
            )
            mock_cal.return_value = cal_inst
            result = get_ff_factors(config)

        assert list(result.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"]

    def test_index_is_tz_naive(self, tmp_path):
        config = _make_config(tmp_path)
        fetch_dates = pd.date_range("2020-01-02", "2020-12-31", freq="B")
        ff5 = _make_raw_ff5(fetch_dates)
        umd = _make_raw_umd(fetch_dates)

        with patch("residrev.factors.pdr.get_data_famafrench") as mock_pdr, \
             patch("residrev.factors.mcal.get_calendar") as mock_cal:
            mock_pdr.side_effect = lambda name, **kw: {0: ff5} if "5_Factors" in name else {0: umd}
            cal_inst = MagicMock()
            cal_inst.valid_days.return_value = pd.DatetimeIndex(
                fetch_dates, dtype="datetime64[ns, UTC]"
            )
            mock_cal.return_value = cal_inst
            result = get_ff_factors(config)

        assert result.index.tz is None


# ---------------------------------------------------------------------------
# get_sector_map
# ---------------------------------------------------------------------------

class TestGetSectorMap:
    _MASTER_JSON = {
        "0": {"cik_str": "1234567890", "ticker": "AAPL", "title": "Apple Inc"},
        "1": {"cik_str": "9876543210", "ticker": "XOM", "title": "Exxon Mobil"},
    }

    def test_loads_from_cache(self, tmp_path):
        config = _make_config(tmp_path)
        cached = {"AAPL": "Business Equipment", "XOM": "Energy"}
        cache_path = os.path.join(str(tmp_path), "sector_map.json")
        with open(cache_path, "w") as f:
            json.dump(cached, f)

        with patch("residrev.factors.requests.Session") as mock_sess:
            result = get_sector_map(["AAPL", "XOM"], config)
            mock_sess.assert_not_called()

        assert result == cached

    def test_maps_correctly_with_mocked_edgar(self, tmp_path):
        config = _make_config(tmp_path)

        master_resp = MagicMock()
        master_resp.json.return_value = self._MASTER_JSON

        # AAPL → SIC 3674 (semiconductors → Business Equipment)
        aapl_resp = MagicMock()
        aapl_resp.json.return_value = {"sic": "3674"}

        # XOM → SIC 1311 (crude oil → Energy)
        xom_resp = MagicMock()
        xom_resp.json.return_value = {"sic": "1311"}

        session_inst = MagicMock()
        session_inst.get.side_effect = [master_resp, aapl_resp, xom_resp]

        with patch("residrev.factors.requests.Session", return_value=session_inst), \
             patch("residrev.factors.time.sleep"):
            result = get_sector_map(["AAPL", "XOM"], config)

        assert result["AAPL"] == "Business Equipment"
        assert result["XOM"] == "Energy"

    def test_assigns_other_for_unknown_ticker(self, tmp_path):
        config = _make_config(tmp_path)

        master_resp = MagicMock()
        master_resp.json.return_value = {}  # empty — no tickers

        session_inst = MagicMock()
        session_inst.get.return_value = master_resp

        with patch("residrev.factors.requests.Session", return_value=session_inst), \
             patch("residrev.factors.time.sleep"):
            result = get_sector_map(["UNKNOWN"], config)

        assert result["UNKNOWN"] == "Other"

    def test_saves_json_cache(self, tmp_path):
        config = _make_config(tmp_path)

        master_resp = MagicMock()
        master_resp.json.return_value = self._MASTER_JSON

        aapl_resp = MagicMock()
        aapl_resp.json.return_value = {"sic": "3674"}

        session_inst = MagicMock()
        session_inst.get.side_effect = [master_resp, aapl_resp]

        with patch("residrev.factors.requests.Session", return_value=session_inst), \
             patch("residrev.factors.time.sleep"):
            get_sector_map(["AAPL"], config)

        cache_path = os.path.join(str(tmp_path), "sector_map.json")
        assert os.path.exists(cache_path)
        with open(cache_path) as f:
            saved = json.load(f)
        assert saved["AAPL"] == "Business Equipment"


# ---------------------------------------------------------------------------
# get_factor_cov
# ---------------------------------------------------------------------------

class TestGetFactorCov:
    def _make_factors(self, n: int = 300) -> pd.DataFrame:
        dates = pd.date_range("2019-01-01", periods=n, freq="B")
        rng = np.random.default_rng(42)
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
        return pd.DataFrame(rng.normal(0, 0.01, (n, len(cols))), index=dates, columns=cols)

    def test_correct_shape(self):
        factors = self._make_factors(300)
        as_of = factors.index[250]
        result = get_factor_cov(factors, as_of)
        assert result is not None
        assert result.shape == (6, 6)

    def test_excludes_as_of_date(self):
        factors = self._make_factors(300)
        as_of = factors.index[250]
        # The row at as_of index must NOT be included
        result_base = get_factor_cov(factors, as_of)

        # Perturb the as_of row to an extreme value — should not affect result
        factors_perturbed = factors.copy()
        factors_perturbed.loc[as_of] = 9999.0
        result_perturbed = get_factor_cov(factors_perturbed, as_of)

        assert result_base is not None
        assert result_perturbed is not None
        np.testing.assert_array_almost_equal(result_base, result_perturbed)

    def test_returns_none_when_too_few_obs(self):
        factors = self._make_factors(50)  # only 50 rows total
        as_of = factors.index[-1]
        result = get_factor_cov(factors, as_of, window=252)
        assert result is None

    def test_uses_window_rows(self):
        factors = self._make_factors(300)
        as_of = factors.index[200]
        # window=100 (>=60 min) → should use rows [100:200]
        result = get_factor_cov(factors, as_of, window=100)
        expected = factors.iloc[100:200].cov().values
        assert result is not None
        np.testing.assert_array_almost_equal(result, expected)
