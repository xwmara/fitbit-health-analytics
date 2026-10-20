"""Statistical and machine learning analyses used by the app.

The functions here have no Streamlit code, so they can be tested on their own.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# --- Outliers ---

def iqr_bounds(series, k=1.5):
    """Return the quartiles, the IQR and the outlier bounds (Q1 - k*IQR, Q3 + k*IQR)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": q1 - k * iqr, "upper": q3 + k * iqr}


def find_outliers(df, column):
    """Return the rows of `df` whose `column` value lies outside the IQR bounds, plus the bounds."""
    bounds = iqr_bounds(df[column].dropna())
    is_outlier = (df[column] < bounds["lower"]) | (df[column] > bounds["upper"])
    return df[is_outlier], bounds


# --- Aggregations ---

def average_steps_by_weekday(daily):
    """Return the mean daily steps for each weekday, ordered Monday to Sunday."""
    day_of_week = daily["ActivityDate"].dt.day_name().rename("DayOfWeek")
    return (
        daily["TotalSteps"].groupby(day_of_week).mean()
        .reindex(WEEKDAY_ORDER)
        .reset_index()
    )


# --- Encoding and scaling ---

def lifestyle_category(steps):
    """Map a daily step count to a lifestyle category (thresholds: 5,000 and 10,000 steps)."""
    if steps < 5000:
        return "Sedentary"
    elif steps < 10000:
        return "Moderately active"
    return "Active"


def add_lifestyle_encoding(df):
    """Return a copy of `df` with `LifestyleCategory` and its label-encoded `LifestyleEncoded`."""
    encoded = df.copy()
    encoded["LifestyleCategory"] = encoded["TotalSteps"].apply(lifestyle_category)
    encoded["LifestyleEncoded"] = LabelEncoder().fit_transform(encoded["LifestyleCategory"])
    return encoded


def standardize(X):
    """Scale every column of `X` to mean 0 and standard deviation 1."""
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)


# --- Clustering ---

def kmeans_clusters(X, n_clusters=3, random_state=42):
    """Standardize `X` and return the K-Means cluster label of every row."""
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(X_scaled)


CLUSTER_NAMES = ["Low activity", "Moderate activity", "Very active"]


def name_clusters(cluster_means, by="TotalSteps", names=CLUSTER_NAMES):
    """Map each cluster label to a descriptive name, ranking clusters by their mean of `by`.

    K-Means numbers its clusters arbitrarily, so names must come from the cluster profiles:
    the cluster with the fewest average steps gets the first name, and so on.
    """
    if len(cluster_means) != len(names):
        raise ValueError(f"Expected {len(names)} clusters, got {len(cluster_means)}")
    ranked = cluster_means[by].sort_values().index
    return dict(zip(ranked, names))


# --- Logistic regression ---

def healthy_lifestyle_label(df):
    """Return 1 for days with >= 8,000 steps, < 1,000 sedentary minutes and >= 420 minutes asleep, else 0."""
    is_healthy = (
        (df["TotalSteps"] >= 8000)
        & (df["SedentaryMinutes"] < 1000)
        & (df["TotalMinutesAsleep"] >= 420)
    )
    return pd.Series(np.where(is_healthy, 1, 0), index=df.index, name="HealthyLifestyle")


@dataclass
class ClassificationResult:
    model: LogisticRegression
    y_test: pd.Series
    y_pred: np.ndarray
    accuracy: float
    baseline_accuracy: float  # accuracy of always predicting the most common training class
    precision: float
    recall: float
    f1: float
    roc_auc: float  # NaN if the test set contains only one class
    confusion_matrix: np.ndarray
    report: str
    coefficients: pd.DataFrame  # columns: Feature, Importance (sorted descending)


def split_train_test(X, y, groups=None, test_size=0.3, random_state=42):
    """Split into train and test sets.

    With `groups` (e.g. user Ids), every group lands entirely in one set, so the model is
    tested on users it has never seen. Without groups, the split is stratified by `y` so
    both sets keep the same class balance.
    """
    if groups is None:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def train_logistic_regression(X, y, groups=None, test_size=0.3, random_state=42,
                              class_weight="balanced"):
    """Split the data, standardize it, fit a logistic regression and evaluate it on the test set.

    `class_weight="balanced"` gives the rare class more weight, so the model does not learn
    to always predict the common class.
    """
    X_train, X_test, y_train, y_test = split_train_test(X, y, groups, test_size, random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight=class_weight)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    majority_class = y_train.mode()[0]
    both_classes_in_test = y_test.nunique() == 2

    coefficients = pd.DataFrame({"Feature": list(X.columns), "Importance": model.coef_[0]})
    coefficients = coefficients.sort_values(by="Importance", ascending=False)

    return ClassificationResult(
        model=model,
        y_test=y_test,
        y_pred=y_pred,
        accuracy=accuracy_score(y_test, y_pred),
        baseline_accuracy=accuracy_score(y_test, np.full(len(y_test), majority_class)),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba) if both_classes_in_test else float("nan"),
        confusion_matrix=confusion_matrix(y_test, y_pred, labels=[0, 1]),
        report=classification_report(y_test, y_pred, labels=[0, 1], zero_division=0),
        coefficients=coefficients,
    )


# --- Linear regression ---

def fit_ols(X, y):
    """Fit an OLS regression of `y` on `X` (with an intercept); return the fitted model and the design matrix."""
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    return model, X_with_const


def standardized_coefficients(model, X, y):
    """Return a table of raw and standardized OLS coefficients, sorted by effect size.

    A standardized coefficient (beta * sd(x) / sd(y)) is the change in `y`, in standard
    deviations, when `x` grows by one standard deviation. Unlike raw coefficients, these
    can be compared across variables measured in different units (steps vs. minutes).
    """
    features = list(X.columns)
    table = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.params[features].values,
        "StdCoefficient": (model.params[features] * X.std() / y.std()).values,
        "PValue": model.pvalues[features].values,
    })
    return table.sort_values("StdCoefficient", key=abs, ascending=False, ignore_index=True)
