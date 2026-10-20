"""Shared colours and Matplotlib style, so every chart in the app looks the same."""

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Palette (also used for Streamlit's built-in charts in .streamlit/config.toml)
PRIMARY = "#0E7C86"   # deep teal
ACCENT = "#E76F51"    # coral
AMBER = "#E9A23B"
BLUE = "#5B7DB1"
GREY = "#8D99AE"
LIGHT_GREY = "#E5E7EB"
LIGHT_TEAL = "#A7D3D6"
TEXT = "#1F2933"
MUTED = "#6B7280"

# Fixed colour per cluster profile, so a profile keeps its colour in every chart
CLUSTER_COLORS = {"Low activity": AMBER, "Moderate activity": BLUE, "Very active": PRIMARY}

# Readable names for the dataset columns, used in tables and chart axes
LABELS = {
    "TotalSteps": "Steps",
    "TotalDistance": "Distance (km)",
    "Calories": "Calories (kcal)",
    "SedentaryMinutes": "Sedentary (min)",
    "VeryActiveMinutes": "Very active (min)",
    "FairlyActiveMinutes": "Fairly active (min)",
    "LightlyActiveMinutes": "Lightly active (min)",
    "TotalMinutesAsleep": "Sleep (min)",
    "TotalTimeInBed": "Time in bed (min)",
    "TotalSleepRecords": "Sleep records",
}


def label(column):
    """Return the readable name of a column (or the column itself if it has none)."""
    return LABELS.get(column, column)


def readable(df):
    """Return a copy of `df` with readable column names."""
    return df.rename(columns=LABELS)


def apply_style():
    """Set a clean, consistent Matplotlib style: light grid, no top/right borders, muted labels."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.dpi": 110,
        "axes.facecolor": "white",
        "axes.edgecolor": "#C9CED6",
        "axes.labelcolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#ECEEF1",
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.prop_cycle": plt.cycler(color=[PRIMARY, ACCENT, AMBER, BLUE, GREY]),
    })


def thousands(ax, axis="both"):
    """Show axis tick labels with thousands separators (5,000 instead of 5000)."""
    formatter = FuncFormatter(lambda value, _: f"{value:,.0f}")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)
