"""Scaffold smoke test — verifies the test suite is discoverable."""


def test_environment():
    """Imports that must be available in the venv."""
    import pandas  # noqa: F401
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import cvxpy  # noqa: F401
    import yfinance  # noqa: F401
    import statsmodels  # noqa: F401
    import pyarrow  # noqa: F401
    import dotenv  # noqa: F401
