from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

import pandas as pd

ERCOT_LOCAL_TIMEZONE = ZoneInfo("America/Chicago")
UTC_OFFSET_BY_SUFFIX = {
    "DST": timezone(timedelta(hours=-5)),
    "ST": timezone(timedelta(hours=-6)),
}


def load_dataset(config: dict, base_dir: Path | None = None) -> pd.DataFrame:
    data_config = config["data"]
    base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    input_path = base_dir / data_config["input_path"]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {input_path}. Update config.json before training."
        )

    df = _read_tabular_data(input_path, data_config)
    timestamp_col = data_config["timestamp_col"]
    target_col = data_config["target_col"]
    feature_cols = data_config.get("feature_cols", [])

    required_columns = [timestamp_col, target_col, *feature_cols]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    df = df[required_columns].copy()
    df[timestamp_col] = parse_timestamp_column(df[timestamp_col])
    df = df.sort_values(timestamp_col).drop_duplicates(subset=timestamp_col).reset_index(drop=True)

    freq = data_config.get("freq")
    if freq:
        df = df.set_index(timestamp_col).asfreq(freq).reset_index()

    numeric_columns = [target_col, *feature_cols]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def parse_timestamp_column(timestamp_series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(timestamp_series):
        return pd.to_datetime(timestamp_series, errors="raise")

    raw = timestamp_series.astype("string").str.strip()
    parsed_values = []
    bad_examples = []

    for value in raw:
        if value is pd.NA or value is None or value == "":
            parsed_values.append(pd.NaT)
            continue

        try:
            parsed_values.append(_parse_single_timestamp(value))
        except ValueError:
            bad_examples.append(value)
            parsed_values.append(pd.NaT)

    parsed = pd.Series(pd.array(parsed_values, dtype="datetime64[ns, America/Chicago]"))
    parsed.index = timestamp_series.index
    parsed.name = timestamp_series.name

    if parsed.isna().any():
        examples = list(dict.fromkeys(bad_examples))[:5]
        raise ValueError(f"Failed to parse timestamps. Example values: {examples}")

    return parsed


def _read_tabular_data(input_path: Path, data_config: dict) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".xlsx", ".xls"}:
        return _read_excel(input_path, data_config.get("sheet_name", 0))
    if suffix == ".zip":
        return _read_from_zip(input_path, data_config)
    raise ValueError(f"Unsupported dataset format: {input_path.suffix}")


def _read_from_zip(zip_path: Path, data_config: dict) -> pd.DataFrame:
    archive_member = data_config.get("archive_member")
    supported_suffixes = (".csv", ".xlsx", ".xls")

    with ZipFile(zip_path) as archive:
        if archive_member:
            member_name = archive_member
        else:
            candidates = [
                name for name in archive.namelist() if name.lower().endswith(supported_suffixes)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "Zip archive must contain exactly one supported data file, "
                    "or config['data']['archive_member'] must be set."
                )
            member_name = candidates[0]

        with archive.open(member_name) as handle:
            payload = BytesIO(handle.read())

    member_suffix = Path(member_name).suffix.lower()
    if member_suffix == ".csv":
        return pd.read_csv(payload)
    if member_suffix in {".xlsx", ".xls"}:
        return _read_excel(payload, data_config.get("sheet_name", 0))
    raise ValueError(f"Unsupported archive member format: {member_suffix}")


def _read_excel(source, sheet_name):
    return pd.read_excel(source, sheet_name=sheet_name)


def _parse_single_timestamp(value: str):
    raw_value = value.strip()
    suffix = None

    for candidate in (" DST", " ST"):
        if raw_value.endswith(candidate):
            suffix = candidate.strip()
            raw_value = raw_value[: -len(candidate)].strip()
            break

    base_dt = datetime.strptime(raw_value.replace("24:00", "00:00"), "%m/%d/%Y %H:%M")
    if "24:00" in raw_value:
        base_dt += timedelta(days=1)

    if suffix is None:
        return base_dt.replace(tzinfo=ERCOT_LOCAL_TIMEZONE)

    aware_with_offset = base_dt.replace(tzinfo=UTC_OFFSET_BY_SUFFIX[suffix])
    return aware_with_offset.astimezone(ERCOT_LOCAL_TIMEZONE)
