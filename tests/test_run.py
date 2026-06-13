"""Tests for residrev/run.py."""

from __future__ import annotations

import json
import os
import textwrap
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from residrev.config import Config
from residrev.run import build_config, get_tickers, log_trial, save_outputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    """Return a SimpleNamespace mimicking argparse.Namespace."""
    from types import SimpleNamespace

    defaults = {
        "start_date": None,
        "end_date": None,
        "universe_size": None,
        "signal_k": None,
        "gamma": None,
        "lam_to": None,
        "output_dir": None,
        "tickers_file": None,
        "eda": False,
        "no_checklist": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_result():
    """Minimal BacktestResult-like object for save_outputs tests."""
    from residrev.backtest import BacktestResult

    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    tickers = ["AAPL", "MSFT"]
    pnl = pd.Series(np.random.randn(5), index=dates)
    positions = pd.DataFrame(np.random.randn(5, 2), index=dates, columns=tickers)
    factor_exposures = pd.DataFrame(
        np.random.randn(5, 3), index=dates, columns=["Mkt-RF", "SMB", "HML"]
    )
    ic_series = pd.Series(np.random.randn(5), index=dates)

    return BacktestResult(
        pnl=pnl,
        gross_pnl=pnl * 1.1,
        positions=positions,
        turnover=pd.Series(np.abs(pnl), index=dates),
        costs_bps=pd.Series(np.abs(pnl) * 2, index=dates),
        factor_exposures=factor_exposures,
        ic_series=ic_series,
        meta={"start_date": "2023-01-01", "end_date": "2023-01-06"},
    )


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def test_no_args_returns_defaults(self):
        args = _make_args()
        cfg = build_config(args)
        defaults = Config()
        assert cfg == defaults

    def test_signal_k_override(self):
        args = _make_args(signal_k=3)
        cfg = build_config(args)
        assert cfg.signal_k == 3
        assert cfg.start_date == Config().start_date  # others unchanged

    def test_start_date_override(self):
        args = _make_args(start_date="2020-01-01")
        cfg = build_config(args)
        assert cfg.start_date == "2020-01-01"

    def test_end_date_override(self):
        args = _make_args(end_date="2022-12-31")
        cfg = build_config(args)
        assert cfg.end_date == "2022-12-31"

    def test_universe_size_override(self):
        args = _make_args(universe_size=500)
        cfg = build_config(args)
        assert cfg.universe_size == 500

    def test_gamma_override(self):
        args = _make_args(gamma=2.5)
        cfg = build_config(args)
        assert cfg.gamma == 2.5

    def test_lam_to_override(self):
        args = _make_args(lam_to=5.0)
        cfg = build_config(args)
        assert cfg.lam_to == 5.0

    def test_output_dir_override(self):
        args = _make_args(output_dir="/tmp/mydata")
        cfg = build_config(args)
        assert cfg.data_dir == "/tmp/mydata"

    def test_multiple_overrides_independent(self):
        args = _make_args(signal_k=3, gamma=2.0)
        cfg = build_config(args)
        assert cfg.signal_k == 3
        assert cfg.gamma == 2.0
        assert cfg.universe_size == Config().universe_size  # untouched

    def test_none_values_not_overridden(self):
        """Passing None for all args must not change any field."""
        args = _make_args()
        cfg = build_config(args)
        for f in Config.__dataclass_fields__:
            assert getattr(cfg, f) == getattr(Config(), f)


# ---------------------------------------------------------------------------
# get_tickers
# ---------------------------------------------------------------------------

class TestGetTickers:
    def test_reads_from_file(self, tmp_path):
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL\nMSFT\nGOOGL\n")
        args = _make_args(tickers_file=str(ticker_file))
        result = get_tickers(args)
        assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_uppercases_tickers(self, tmp_path):
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("aapl\nmsft\n")
        args = _make_args(tickers_file=str(ticker_file))
        result = get_tickers(args)
        assert result == ["AAPL", "MSFT"]

    def test_deduplicates_tickers(self, tmp_path):
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL\nMSFT\nAAPL\n")
        args = _make_args(tickers_file=str(ticker_file))
        result = get_tickers(args)
        assert result.count("AAPL") == 1
        assert len(result) == 2

    def test_strips_whitespace(self, tmp_path):
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("  AAPL  \n  MSFT\n")
        args = _make_args(tickers_file=str(ticker_file))
        result = get_tickers(args)
        assert result == ["AAPL", "MSFT"]

    def test_skips_blank_lines(self, tmp_path):
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL\n\nMSFT\n\n")
        args = _make_args(tickers_file=str(ticker_file))
        result = get_tickers(args)
        assert result == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# log_trial
# ---------------------------------------------------------------------------

class TestLogTrial:
    def test_appends_to_file(self, tmp_path):
        log_path = str(tmp_path / "trials.jsonl")
        cfg = Config(trials_log=log_path)
        summary = {"net_sharpe": 1.5}
        log_trial(summary, cfg, "run_001")
        log_trial(summary, cfg, "run_002")
        lines = open(log_path).readlines()
        assert len(lines) == 2

    def test_creates_parent_directory(self, tmp_path):
        log_path = str(tmp_path / "nested" / "dir" / "trials.jsonl")
        cfg = Config(trials_log=log_path)
        log_trial({"x": 1}, cfg, "run_abc")
        assert os.path.exists(log_path)

    def test_each_line_is_valid_json(self, tmp_path):
        log_path = str(tmp_path / "trials.jsonl")
        cfg = Config(trials_log=log_path)
        log_trial({"a": 1}, cfg, "r1")
        log_trial({"b": 2}, cfg, "r2")
        lines = open(log_path).readlines()
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_entry_contains_run_id_and_summary(self, tmp_path):
        log_path = str(tmp_path / "trials.jsonl")
        cfg = Config(trials_log=log_path)
        log_trial({"net_sharpe": 2.0}, cfg, "run_xyz")
        entry = json.loads(open(log_path).readline())
        assert entry["run_id"] == "run_xyz"
        assert entry["summary"]["net_sharpe"] == 2.0

    def test_does_not_overwrite_existing_entries(self, tmp_path):
        log_path = str(tmp_path / "trials.jsonl")
        cfg = Config(trials_log=log_path)
        log_trial({"n": 1}, cfg, "first")
        log_trial({"n": 2}, cfg, "second")
        lines = open(log_path).readlines()
        ids = [json.loads(l)["run_id"] for l in lines]
        assert "first" in ids and "second" in ids


# ---------------------------------------------------------------------------
# save_outputs
# ---------------------------------------------------------------------------

class TestSaveOutputs:
    def test_creates_run_directory(self, tmp_path):
        cfg = Config(data_dir=str(tmp_path))
        result = _make_result()
        save_outputs(result, {"x": 1}, "run_001", cfg)
        assert os.path.isdir(tmp_path / "results" / "run_001")

    def test_all_five_files_exist(self, tmp_path):
        cfg = Config(data_dir=str(tmp_path))
        result = _make_result()
        save_outputs(result, {"x": 1}, "run_001", cfg)
        out = tmp_path / "results" / "run_001"
        for fname in ["summary.json", "pnl.parquet", "positions.parquet",
                      "factor_exposures.parquet", "ic_series.parquet"]:
            assert (out / fname).exists(), f"{fname} missing"

    def test_summary_json_round_trips(self, tmp_path):
        cfg = Config(data_dir=str(tmp_path))
        result = _make_result()
        summary = {"net_sharpe": 1.23, "label": "test"}
        save_outputs(result, summary, "run_001", cfg)
        loaded = json.loads((tmp_path / "results" / "run_001" / "summary.json").read_text())
        assert loaded == summary

    def test_pnl_parquet_has_correct_columns(self, tmp_path):
        cfg = Config(data_dir=str(tmp_path))
        result = _make_result()
        save_outputs(result, {}, "run_001", cfg)
        pnl_df = pd.read_parquet(tmp_path / "results" / "run_001" / "pnl.parquet")
        assert set(pnl_df.columns) == {"gross_pnl", "net_pnl", "costs_bps"}

    def test_ic_series_parquet_has_ic_column(self, tmp_path):
        cfg = Config(data_dir=str(tmp_path))
        result = _make_result()
        save_outputs(result, {}, "run_001", cfg)
        ic_df = pd.read_parquet(tmp_path / "results" / "run_001" / "ic_series.parquet")
        assert "ic" in ic_df.columns


# ---------------------------------------------------------------------------
# run() — mocked pipeline
# ---------------------------------------------------------------------------

_RUN_PATCH_BASE = "residrev.run"


def _make_mock_result():
    return _make_result()


class TestRunFunction:
    @patch(f"{_RUN_PATCH_BASE}.run_backtest")
    @patch(f"{_RUN_PATCH_BASE}.summarize")
    @patch(f"{_RUN_PATCH_BASE}.build_signal", return_value=(MagicMock(), MagicMock()))
    @patch(f"{_RUN_PATCH_BASE}.rolling_residuals", return_value=(MagicMock(), {}, MagicMock()))
    @patch(f"{_RUN_PATCH_BASE}.build_return_panel", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.get_liquid_universe", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.compute_adv", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.get_ff_factors", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.pull_prices", return_value={})
    @patch(f"{_RUN_PATCH_BASE}.get_sector_map", return_value={})
    @patch(f"{_RUN_PATCH_BASE}.compute_amihud", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.get_vix", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.compute_vix_regime", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.corwin_schultz_spread", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.compute_realized_vol", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.compute_idio_vol", return_value=MagicMock())
    def test_run_backtest_called_exactly_once(
        self, mock_idio_vol, mock_rvol, mock_cs, mock_vix_regime, mock_vix, mock_amihud,
        mock_sector, mock_prices, mock_ff, mock_adv, mock_univ, mock_returns,
        mock_resid, mock_signal, mock_summarize, mock_backtest,
    ):
        mock_backtest.return_value = _make_mock_result()
        mock_summarize.return_value = {"net_sharpe": 1.0}

        from residrev.run import run
        cfg = Config()
        result, summary = run(cfg, ["AAPL", "MSFT"])

        mock_backtest.assert_called_once()
        assert summary == {"net_sharpe": 1.0}

    @patch(f"{_RUN_PATCH_BASE}.run_backtest")
    @patch(f"{_RUN_PATCH_BASE}.summarize")
    @patch(f"{_RUN_PATCH_BASE}.build_signal", return_value=(MagicMock(), MagicMock()))
    @patch(f"{_RUN_PATCH_BASE}.rolling_residuals", return_value=(MagicMock(), {}, MagicMock()))
    @patch(f"{_RUN_PATCH_BASE}.build_return_panel", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.get_liquid_universe", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.compute_adv", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.get_ff_factors", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.pull_prices", return_value={})
    @patch(f"{_RUN_PATCH_BASE}.get_sector_map", return_value={})
    @patch(f"{_RUN_PATCH_BASE}.compute_amihud", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.get_vix", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.compute_vix_regime", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.corwin_schultz_spread", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.compute_realized_vol", return_value=MagicMock())
    @patch(f"{_RUN_PATCH_BASE}.compute_idio_vol", return_value=MagicMock())
    def test_run_returns_backtest_result_and_summary(
        self, mock_idio_vol, mock_rvol, mock_cs, mock_vix_regime, mock_vix, mock_amihud,
        mock_sector, mock_prices, mock_ff, mock_adv, mock_univ, mock_returns,
        mock_resid, mock_signal, mock_summarize, mock_backtest,
    ):
        expected_result = _make_mock_result()
        mock_backtest.return_value = expected_result
        mock_summarize.return_value = {"gross_sharpe": 2.0}

        from residrev.run import run
        result, summary = run(Config(), ["AAPL"])

        assert result is expected_result
        assert summary["gross_sharpe"] == 2.0


# ---------------------------------------------------------------------------
# main() — import and --help
# ---------------------------------------------------------------------------

class TestMain:
    def test_help_exits_cleanly(self):
        """python -m residrev.run --help must exit 0."""
        import subprocess
        proc = subprocess.run(
            ["python", "-m", "residrev.run", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert proc.returncode == 0
        assert "usage" in proc.stdout.lower()

    def test_importable(self):
        from residrev.run import run, main  # noqa: F401
