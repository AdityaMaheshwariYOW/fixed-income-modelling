import pandas as pd
import tempfile
from scripts.utils import read_and_parse_dates

def test_date_parsing():
    df = pd.DataFrame({'date': ['2023-01-01']})

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        result = read_and_parse_dates(tmp.name, 'date')

    assert pd.api.types.is_datetime64_any_dtype(result['date'])

