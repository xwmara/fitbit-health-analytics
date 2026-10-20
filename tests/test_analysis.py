import numpy as np
import pandas as pd
import pytest

from fitbit_analytics import analysis


# --- Outliers ---

def test_iqr_bounds():
    bounds = analysis.iqr_bounds(pd.Series([1, 2, 3, 4, 5, 6, 7, 8]))
    assert bounds["q1"] == pytest.approx(2.75)
    assert bounds["q3"] == pytest.approx(6.25)
    assert bounds["iqr"] == pytest.approx(3.5)
    assert bounds["lower"] == pytest.approx(-2.5)
    assert bounds["upper"] == pytest.approx(11.5)


def test_find_outliers_returns_only_values_outside_bounds():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 100, None]})
    outliers, bounds = analysis.find_outliers(df, "x")
    assert outliers["x"].tolist() == [100]
    assert bounds["upper"] < 100


# --- Aggregations ---

def test_average_steps_by_weekday_is_ordered_monday_to_sunday():
    # 2016-04-11 was a Monday; two Mondays and one Tuesday
    daily = pd.DataFrame({
        "ActivityDate": pd.to_datetime(["2016-04-11", "2016-04-18", "2016-04-12"]),
        "TotalSteps": [1000, 3000, 5000],
    })
    result = analysis.average_steps_by_weekday(daily)
    assert result["DayOfWeek"].tolist() == analysis.WEEKDAY_ORDER
    assert result.loc[0, "TotalSteps"] == 2000  # Monday
    assert result.loc[1, "TotalSteps"] == 5000  # Tuesday
    assert result["TotalSteps"].iloc[2:].isna().all()  # no data for the other days
    assert "DayOfWeek" not in daily.columns  # input is not modified


# --- Encoding and scaling ---

@pytest.mark.parametrize("steps, expected", [
    (0, "Sedentary"),
    (4999, "Sedentary"),
    (5000, "Moderately active"),
    (9999, "Moderately active"),
    (10000, "Active"),
])
def test_lifestyle_category_thresholds(steps, expected):
    assert analysis.lifestyle_category(steps) == expected


def test_add_lifestyle_encoding():
    df = pd.DataFrame({"TotalSteps": [12000, 7000, 3000]})
    encoded = analysis.add_lifestyle_encoding(df)
    assert encoded["LifestyleCategory"].tolist() == ["Active", "Moderately active", "Sedentary"]
    # LabelEncoder sorts the labels alphabetically
    assert encoded["LifestyleEncoded"].tolist() == [0, 1, 2]
    assert "LifestyleCategory" not in df.columns


def test_standardize_gives_mean_zero_and_unit_std():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 40.0, 80.0]})
    scaled = analysis.standardize(X)
    assert list(scaled.columns) == ["a", "b"]
    assert np.allclose(scaled.mean(), 0)
    assert np.allclose(scaled.std(ddof=0), 1)


# --- Clustering ---

def test_kmeans_clusters_separates_distinct_groups():
    X = pd.DataFrame({
        "steps": [1000, 1100, 1200, 15000, 15100, 15200],
        "calories": [1500, 1550, 1600, 3500, 3550, 3600],
    })
    labels = analysis.kmeans_clusters(X, n_clusters=2)
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]


def test_kmeans_clusters_is_reproducible():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    assert (analysis.kmeans_clusters(X) == analysis.kmeans_clusters(X)).all()


def test_name_clusters_ranks_by_average_steps():
    # Cluster labels deliberately out of order: label 2 has the fewest steps
    means = pd.DataFrame({"TotalSteps": [8000, 14000, 4000]}, index=[0, 1, 2])
    assert analysis.name_clusters(means) == {2: "Low activity", 0: "Moderate activity", 1: "Very active"}


def test_name_clusters_rejects_wrong_number_of_clusters():
    means = pd.DataFrame({"TotalSteps": [4000, 8000]})
    with pytest.raises(ValueError):
        analysis.name_clusters(means)


# --- Logistic regression ---

def test_healthy_lifestyle_label_thresholds():
    df = pd.DataFrame({
        "TotalSteps":         [8000, 7999, 8000, 8000],
        "SedentaryMinutes":   [999,  999,  1000, 999],
        "TotalMinutesAsleep": [420,  420,  420,  419],
    })
    assert analysis.healthy_lifestyle_label(df).tolist() == [1, 0, 0, 0]


def test_train_logistic_regression_on_separable_data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    X = pd.DataFrame({"signal": x, "noise": rng.normal(size=200)})
    y = pd.Series((x > 0).astype(int))

    result = analysis.train_logistic_regression(X, y)

    assert len(result.y_test) == 60  # 30% of 200
    assert result.accuracy > 0.9
    assert result.recall > 0.9
    assert result.roc_auc > 0.95
    assert result.confusion_matrix.shape == (2, 2)
    assert result.coefficients["Feature"].iloc[0] == "signal"
    assert result.coefficients["Importance"].is_monotonic_decreasing


def test_baseline_accuracy_is_share_of_majority_class():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": rng.normal(size=300)})
    y = pd.Series((rng.random(300) < 0.1).astype(int))  # ~10% positives

    result = analysis.train_logistic_regression(X, y)

    assert result.baseline_accuracy == pytest.approx(1 - result.y_test.mean())


def test_split_train_test_keeps_each_group_in_one_set():
    X = pd.DataFrame({"x": range(100)})
    y = pd.Series([0, 1] * 50)
    groups = pd.Series(np.repeat(range(10), 10))  # 10 users with 10 days each

    X_train, X_test, _, _ = analysis.split_train_test(X, y, groups=groups)

    train_users = set(groups[X_train.index])
    test_users = set(groups[X_test.index])
    assert train_users.isdisjoint(test_users)
    assert len(test_users) == 3  # 30% of 10 users


def test_split_train_test_without_groups_is_stratified():
    X = pd.DataFrame({"x": range(100)})
    y = pd.Series([1] * 20 + [0] * 80)

    _, _, y_train, y_test = analysis.split_train_test(X, y)

    assert y_train.mean() == pytest.approx(0.2)
    assert y_test.mean() == pytest.approx(0.2)


# --- Linear regression ---

def test_fit_ols_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x1": rng.normal(size=100), "x2": rng.normal(size=100)})
    y = 3 + 2 * X["x1"] - 1 * X["x2"]

    model, X_with_const = analysis.fit_ols(X, y)

    assert list(X_with_const.columns) == ["const", "x1", "x2"]
    assert model.params["const"] == pytest.approx(3)
    assert model.params["x1"] == pytest.approx(2)
    assert model.params["x2"] == pytest.approx(-1)
    assert model.rsquared == pytest.approx(1)


def test_standardized_coefficients_are_unit_free():
    rng = np.random.default_rng(2)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    # Same underlying effect for both, but x1 is recorded in units 1000 times larger
    X = pd.DataFrame({"big_units": x1 * 1000, "small_units": x2})
    y = x1 + x2 + rng.normal(scale=0.1, size=200)

    model, _ = analysis.fit_ols(X, y)
    table = analysis.standardized_coefficients(model, X, y).set_index("Feature")

    # Raw coefficients differ by ~1000x; standardized ones are nearly equal
    assert table.loc["small_units", "Coefficient"] / table.loc["big_units", "Coefficient"] > 500
    assert table.loc["big_units", "StdCoefficient"] == pytest.approx(
        table.loc["small_units", "StdCoefficient"], rel=0.2
    )
    assert table["StdCoefficient"].abs().is_monotonic_decreasing
