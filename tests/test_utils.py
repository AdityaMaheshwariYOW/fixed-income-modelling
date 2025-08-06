import pandas as pd
from scripts.utils import read_and_parse_dates

def test_date_parsing():
    df = pd.DataFrame({'date': ['2023-01-01']})
    result = read_and_parse_dates(df, 'date')
    assert pd.api.types.is_datetime64_any_dtype(result['date'])

