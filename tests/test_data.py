import pandas as pd
import pytest

from fitbit_analytics import data


@pytest.fixture(scope="module")
def raw():
    return data.load_raw_data()


def test_load_raw_data_reads_both_files(raw):
    daily, sleep = raw
    assert daily.shape == (940, 15)
    assert sleep.shape == (410, 5)  # 413 rows in the file minus 3 exact duplicates
    assert daily["Id"].nunique() == 33
    assert sleep["Id"].nunique() == 24


def test_load_raw_data_parses_dates(raw):
    daily, sleep = raw
    assert pd.api.types.is_datetime64_any_dtype(daily["ActivityDate"])
    assert pd.api.types.is_datetime64_any_dtype(sleep["SleepDay"])


def test_load_data_works_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    daily, sleep, merged = data.load_data()
    assert len(daily) == 940


def test_merge_keeps_every_daily_row():
    daily, _, merged = data.load_data()
    daily_keys = set(zip(daily["Id"], daily["ActivityDateOnly"]))
    merged_keys = set(zip(merged["Id"], merged["ActivityDateOnly"]))
    assert merged_keys == daily_keys


def test_load_raw_data_removes_duplicate_rows(raw):
    daily, sleep = raw
    assert not daily.duplicated().any()
    assert not sleep.duplicated().any()


def test_merge_has_exactly_one_row_per_user_and_day():
    daily, _, merged = data.load_data()
    assert len(merged) == len(daily) == 940
    assert not merged.duplicated(["Id", "ActivityDateOnly"]).any()


def test_merge_leaves_sleep_missing_for_days_without_a_record():
    daily = pd.DataFrame({
        "Id": [1, 1],
        "ActivityDate": pd.to_datetime(["2016-04-12", "2016-04-13"]),
        "TotalSteps": [1000, 2000],
    })
    sleep = pd.DataFrame({
        "Id": [1],
        "SleepDay": pd.to_datetime(["2016-04-12"]),
        "TotalSleepRecords": [1],
        "TotalMinutesAsleep": [400],
        "TotalTimeInBed": [420],
    })
    merged = data.merge_activity_sleep(daily, sleep)
    assert len(merged) == 2
    assert merged.loc[0, "TotalMinutesAsleep"] == 400
    assert pd.isna(merged.loc[1, "TotalMinutesAsleep"])


def test_days_with_sleep_keeps_only_real_sleep_records():
    _, sleep, merged = data.load_data()
    subset = data.days_with_sleep(merged)
    assert len(subset) == len(sleep) == 410
    assert subset["TotalMinutesAsleep"].notna().all()
    assert subset["Id"].nunique() == 24


def test_fill_missing_with_median_fills_only_selected_columns():
    df = pd.DataFrame({"a": [1.0, None, 3.0, 10.0], "b": [None, 2.0, 2.0, 2.0]})
    filled = data.fill_missing_with_median(df, ["a"])
    assert filled["a"].tolist() == [1.0, 3.0, 3.0, 10.0]
    assert filled["b"].isna().sum() == 1


def test_fill_missing_with_median_defaults_to_all_numeric_columns():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 4.0, 6.0], "label": ["x", None, "y"]})
    filled = data.fill_missing_with_median(df)
    assert filled["a"].tolist() == [1.0, 2.0, 3.0]
    assert filled["b"].tolist() == [5.0, 4.0, 6.0]
    assert filled["label"].isna().sum() == 1  # text columns are left alone


def test_fill_missing_with_median_does_not_modify_input():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    data.fill_missing_with_median(df, ["a"])
    assert df["a"].isna().sum() == 1
