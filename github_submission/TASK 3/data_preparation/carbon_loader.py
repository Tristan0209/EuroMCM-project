import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import read_json

REGIONS = {'DE', 'FR', 'ES'}
VALUE = 'carbon_intensity_gco2_kwh'
VALUE_BY_SPECIES = {'CO2': VALUE, 'CO2e': 'carbon_intensity_gco2e_kwh'}


def validate_carbon_metadata(meta):
    species = meta.get('emissions_species')
    units = {'CO2':'gCO2/kWh', 'CO2e':'gCO2e/kWh'}
    if species not in units or meta.get('unit') != units[species]:
        raise ValueError('emissions_species, unit and value column must distinguish CO2 from CO2e; no automatic conversion')
    if not meta.get('source') or not meta.get('boundary'):
        raise ValueError('Explicit source and emissions boundary required')
    method = meta.get('methodology', {})
    if not isinstance(method,dict):
        raise ValueError('methodology must be an object')
    if (method.get('accounting_basis') not in {'production','consumption','marginal','synthetic_test_only'}
            or method.get('scope') not in {'operational','lifecycle','synthetic_test_only'}
            or not method.get('description')):
        raise ValueError('methodology must declare accounting_basis, scope and description')
    if any('REPLACE' in str(value) for value in [meta['source'],meta['boundary'],method['description'],method.get('gwp_reference','')]):
        raise ValueError('Fill actual provider methodology; example placeholders cannot be validated')
    if species == 'CO2e' and (not isinstance(method.get('gwp_horizon_years'),(int,float))
            or not np.isfinite(method['gwp_horizon_years']) or method['gwp_horizon_years'] <= 0 or not method.get('gwp_reference')):
        raise ValueError('CO2e requires positive gwp_horizon_years and gwp_reference')
    return VALUE_BY_SPECIES[species]


def assert_compatible_carbon(actual, forecast):
    a, b = actual.attrs['metadata'], forecast.attrs['metadata']
    fields = ['unit','emissions_species','boundary']
    if any(a.get(k) != b.get(k) for k in fields):
        raise ValueError('Actual/forecast carbon units, species or boundaries differ; cannot mix CO2 with CO2e')
    for k in ['accounting_basis','scope','gwp_horizon_years','gwp_reference']:
        if a['methodology'].get(k) != b['methodology'].get(k):
            raise ValueError('Actual/forecast methodology mismatch: '+k)


def utc_column(series, name):

    if not series.astype(str).str.contains(r'(?:Z|\+00:00)$', regex=True).all():
        raise ValueError(name + ' must explicitly use UTC Z or +00:00')
    try:
        return pd.to_datetime(series, utc=True, format='ISO8601', errors='raise')
    except (ValueError, TypeError):
        return pd.to_datetime(series, utc=True, errors='raise')


def load_carbon(path, metadata_path=None, required_start=None, required_end=None,
                allow_synthetic=False, expected_regions=REGIONS):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Missing hourly carbon data: ' + str(path) + '; no annual factor or synthetic fallback is permitted')
    meta_path = Path(metadata_path) if metadata_path else path.with_suffix('.metadata.json')
    if not meta_path.is_file():
        raise ValueError('Carbon unit/source/boundary metadata required: ' + str(meta_path))
    meta = read_json(meta_path)
    value_column = validate_carbon_metadata(meta)
    kind = meta.get('data_kind')
    if kind not in {'actual', 'forecast', 'synthetic_test_only'}:
        raise ValueError('data_kind must be actual, forecast, or synthetic_test_only')
    if kind == 'synthetic_test_only' and not allow_synthetic:
        raise ValueError('synthetic_test_only is forbidden for real experiments')
    if meta.get('resolution') != 'hourly' or meta.get('timezone') != 'UTC':
        raise ValueError('resolution=hourly and timezone=UTC required')
    f = pd.read_csv(path)
    other_columns = set(VALUE_BY_SPECIES.values()) - {value_column}
    if other_columns & set(f.columns):
        raise ValueError('Ambiguous or mismatched CO2/CO2e value column')
    required = ['datetime_utc','region',value_column] + (['issued_at_utc'] if kind=='forecast' else [])
    if not set(required).issubset(f.columns) or f.empty:
        raise ValueError('Missing carbon columns or empty data')
    if f[required].isna().any().any():
        raise ValueError('Null carbon fields')
    f['datetime_utc'] = utc_column(f.datetime_utc, 'datetime_utc')
    if not (f.datetime_utc == f.datetime_utc.dt.floor('h')).all():
        raise ValueError('Carbon datetime must be on an hourly boundary')
    f[value_column] = pd.to_numeric(f[value_column], errors='raise')
    if not np.isfinite(f[value_column]).all() or not f[value_column].ge(0).all():
        raise ValueError('Carbon intensity must be finite and nonnegative')
    if set(f.region) != set(expected_regions):
        raise ValueError('Carbon regions must exactly match server regions DE/FR/ES')
    keys = ['datetime_utc','region']
    if kind=='forecast':
        f['issued_at_utc'] = utc_column(f.issued_at_utc, 'issued_at_utc')
        if (f.issued_at_utc > f.datetime_utc).any():
            raise ValueError('Forecast issuance must not follow its target hour')
        keys += ['issued_at_utc']
    if f.duplicated(keys).any():
        raise ValueError('Duplicate carbon keys')

    groups = f.groupby('issued_at_utc') if kind=='forecast' else [(None,f)]
    for _, block in groups:
        start, end = block.datetime_utc.min(), block.datetime_utc.max()
        expected = pd.MultiIndex.from_product([pd.date_range(start,end,freq='h'), sorted(expected_regions)])
        actual = pd.MultiIndex.from_frame(block[['datetime_utc','region']])
        if not expected.difference(actual).empty:
            raise ValueError('Hourly carbon series has missing hours/regions')
    if required_start is not None or required_end is not None:
        start = pd.Timestamp(required_start).floor('h') if required_start is not None else f.datetime_utc.min()
        end = pd.Timestamp(required_end).floor('h') if required_end is not None else f.datetime_utc.max()
        expected = pd.MultiIndex.from_product([pd.date_range(start,end,freq='h'), sorted(expected_regions)])
        actual = pd.MultiIndex.from_frame(f[['datetime_utc','region']].drop_duplicates())
        if not expected.difference(actual).empty:
            raise ValueError('Carbon coverage does not cover requests and completion deadlines')
    f.attrs['metadata'] = meta
    f.attrs['value_column'] = value_column
    return f.sort_values(keys).reset_index(drop=True)


def carbon_for_decision(frame, decision_time_utc, target_times_utc):

    if frame.attrs.get('metadata',{}).get('data_kind') != 'forecast':
        raise ValueError('Decision input requires forecast data, never future actuals/synthetic data')
    now = utc_column(pd.Series([decision_time_utc]), 'decision_time_utc').iloc[0]
    targets = utc_column(pd.Series(target_times_utc), 'target_times_utc').dt.floor('h')
    available = frame[frame.issued_at_utc.le(now) & frame.datetime_utc.isin(targets)]
    result = available.sort_values('issued_at_utc').drop_duplicates(['datetime_utc','region'], keep='last')
    expected = pd.MultiIndex.from_product([targets.unique(), sorted(REGIONS)])
    if not expected.difference(pd.MultiIndex.from_frame(result[['datetime_utc','region']])).empty:
        raise ValueError('No available forecast for at least one requested hour/region; do not read future actuals')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args()
    result = load_carbon(args.path,args.metadata)
    print('Validated carbon rows:',len(result),'kind:',result.attrs['metadata']['data_kind'])
