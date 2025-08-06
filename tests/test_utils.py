import pandas as pd
import numpy as np
import tempfile
import os
from scripts.utils import read_and_parse_dates, add_time_column, check_cashflow_consistency

def test_read_and_parse_dates_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w', delete=False) as f:
        f.write("date,value\n2023-01-01,100\n2024-01-01,200\n")
        f.flush()
        result = read_and_parse_dates(f.name, 'date')
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    os.remove(f.name)


def test_add_time_column():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2024-01-01'])
    })
    result = add_time_column(df.copy(), 'date')
    assert 'time' in result.columns
    assert np.isclose(result['time'].iloc[0], 0.0)
    assert np.isclose(result['time'].iloc[1], 1.0, atol=0.01)


def test_check_cashflow_consistency_exact_match(capfd):
    df = pd.DataFrame({
        'company': ['A'],
        'date': ['2023-01-01'],
        'coupon': [1.2],
        'delta_notional': [-20],
        'price': [1],
        'total_cf': [-18.8]
    })
    check_cashflow_consistency(df)
    captured = capfd.readouterr()
    assert "Number of mismatches in cashflow: 0" in captured.out


def test_check_cashflow_consistency_mismatch(capfd):
    df = pd.DataFrame({
        'company': ['A'],
        'date': ['2023-01-01'],
        'coupon': [1.2],
        'delta_notional': [-20],
        'price': [1],
        'total_cf': [-18.0]  # should be -18.8
    })
    check_cashflow_consistency(df)
    captured = capfd.readouterr()
    assert "Number of mismatches in cashflow: 1" in captured.out


def test_read_and_parse_dates_invalid_format():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode='w', delete=False) as f:
        f.write("random text")
        f.flush()
        try:
            read_and_parse_dates(f.name)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        finally:
            os.remove(f.name)
