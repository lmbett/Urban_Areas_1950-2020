#!/usr/bin/env python3
"""
Plot scaling parameters over time using the output CSV from
scaling_decomposition_panel_optionA_updated.py.

Produces time-series plots for:
  - b(t), a(t), A0(t) for free-b fits
  - a_fixed(t), A0_fixed(t) for fixed b = 5/6
All with 95% CI error bars.

Series plotted:
  - All UAs
  - Panel cities
  - Non-panel cities
  - 2020 >50k UAs (when available)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

CSV_FILE = "scaling_decomposition_panel.csv"
OUTDIR = "scaling_parameter_timeseries_plots"
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------------------------------------
# Helper: error bar plotting
# ------------------------------------------------------------
def add_errorbars(ax, x, y, lo, hi, label, color, marker):
    yerr = np.vstack([y - lo, hi - y])
    ax.errorbar(
        x, y, yerr=yerr,
        fmt=marker, markersize=7, capsize=4,
        color=color, label=label, lw=2
    )


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(CSV_FILE)
YEARS = df["year"].values

COL_ALL = "black"
COL_PANEL = "red"
COL_NON = "green"
COL_50K = "blue"
COL_FIXED = "purple"


# ------------------------------------------------------------
# 1. Plot b over time (free-b fits)
# ------------------------------------------------------------
def plot_b_over_time(df):
    fig, ax = plt.subplots(figsize=(8,6))

    # All UAs
    add_errorbars(ax, df["year"], df["b_all"],
                  df["b_all_lo"], df["b_all_hi"],
                  label="All UAs", color=COL_ALL, marker="o")

    # Panel
    add_errorbars(ax, df["year"], df["b_panel"],
                  df["b_panel_lo"], df["b_panel_hi"],
                  label="Panel", color=COL_PANEL, marker="s")

    # Non-panel
    add_errorbars(ax, df["year"], df["b_nonpanel"],
                  df["b_nonpanel_lo"], df["b_nonpanel_hi"],
                  label="Non-panel", color=COL_NON, marker="^")

    # 2020 >50k
    mask_2020 = df["year"] == 2020
    if not df["b_2020_gt50k"].isna().all():
        add_errorbars(
            ax, df["year"][mask_2020], df["b_2020_gt50k"][mask_2020],
            df["b_2020_gt50k_lo"][mask_2020],
            df["b_2020_gt50k_hi"][mask_2020],
            label="2020 (>50k)", color=COL_50K, marker="D"
        )

    # Horizontal line at b = 5/6
    ax.axhline(5/6, color="gray", linewidth=3, alpha=0.5, label="b = 5/6")

    ax.set_title("Urban Scaling Exponent b Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Exponent b")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "b_over_time.png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# 2. Plot a over time (free-b fits)
# ------------------------------------------------------------
def plot_a_over_time(df):
    fig, ax = plt.subplots(figsize=(8,6))

    add_errorbars(ax, df["year"], df["a_all"],
                  df["a_all_lo"], df["a_all_hi"],
                  label="All UAs", color=COL_ALL, marker="o")

    add_errorbars(ax, df["year"], df["a_panel"],
                  df["a_panel_lo"], df["a_panel_hi"],
                  label="Panel", color=COL_PANEL, marker="s")

    add_errorbars(ax, df["year"], df["a_nonpanel"],
                  df["a_nonpanel_lo"], df["a_nonpanel_hi"],
                  label="Non-panel", color=COL_NON, marker="^")

    # 2020 >50k
    mask_2020 = df["year"] == 2020
    if not df["a_2020_gt50k"].isna().all():
        add_errorbars(ax,
                      df["year"][mask_2020], df["a_2020_gt50k"][mask_2020],
                      df["a_2020_gt50k_lo"][mask_2020],
                      df["a_2020_gt50k_hi"][mask_2020],
                      label="2020 (>50k)", color=COL_50K, marker="D")

    ax.set_title("Intercept a Over Time (Free b)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Intercept a")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "a_over_time.png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# 3. Plot A0 over time (free-b fits)
# ------------------------------------------------------------
def plot_A0_over_time(df):
    fig, ax = plt.subplots(figsize=(8,6))

    add_errorbars(ax, df["year"], df["A0_all"],
                  df["A0_all_lo"], df["A0_all_hi"],
                  label="All UAs", color=COL_ALL, marker="o")

    add_errorbars(ax, df["year"], df["A0_panel"],
                  df["A0_panel_lo"], df["A0_panel_hi"],
                  label="Panel", color=COL_PANEL, marker="s")

    add_errorbars(ax, df["year"], df["A0_nonpanel"],
                  df["A0_nonpanel_lo"], df["A0_nonpanel_hi"],
                  label="Non-panel", color=COL_NON, marker="^")

    # 2020 >50k
    mask_2020 = df["year"] == 2020
    if not df["A0_2020_gt50k"].isna().all():
        add_errorbars(ax,
                      df["year"][mask_2020], df["A0_2020_gt50k"][mask_2020],
                      df["A0_2020_gt50k_lo"][mask_2020],
                      df["A0_2020_gt50k_hi"][mask_2020],
                      label="2020 (>50k)", color=COL_50K, marker="D")

    ax.set_title("Baseline Area A₀ Over Time (Free b)")
    ax.set_xlabel("Year")
    ax.set_ylabel("A₀ (hectares)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "A0_over_time.png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# 4. Plot a_fixed over time (b = 5/6)
# ------------------------------------------------------------
def plot_a_fixed_over_time(df):
    fig, ax = plt.subplots(figsize=(8,6))

    add_errorbars(ax, df["year"], df["a_fixed_all"],
                  df["a_fixed_all_lo"], df["a_fixed_all_hi"],
                  label="All UAs (fixed b)", color=COL_FIXED, marker="o")

    add_errorbars(ax, df["year"], df["a_fixed_panel"],
                  df["a_fixed_panel_lo"], df["a_fixed_panel_hi"],
                  label="Panel (fixed b)", color="orange", marker="s")

    add_errorbars(ax, df["year"], df["a_fixed_nonpanel"],
                  df["a_fixed_nonpanel_lo"], df["a_fixed_nonpanel_hi"],
                  label="Non-panel (fixed b)", color="green", marker="^")

    # 2020 >50k
    mask_2020 = df["year"] == 2020
    if not df["a_fixed_2020_gt50k"].isna().all():
        add_errorbars(ax,
                      df["year"][mask_2020], df["a_fixed_2020_gt50k"][mask_2020],
                      df["a_fixed_2020_gt50k_lo"][mask_2020],
                      df["a_fixed_2020_gt50k_hi"][mask_2020],
                      label="2020 (>50k, fixed b)", color=COL_50K, marker="D")

    ax.set_title("Intercept a Over Time (Fixed b = 5/6)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Intercept a (fixed b)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "a_fixed_over_time.png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# 5. Plot A0_fixed over time (b = 5/6)
# ------------------------------------------------------------
def plot_A0_fixed_over_time(df):
    fig, ax = plt.subplots(figsize=(8,6))

    add_errorbars(ax, df["year"], df["A0_fixed_all"],
                  df["A0_fixed_all_lo"], df["A0_fixed_all_hi"],
                  label="All UAs (fixed b)", color=COL_FIXED, marker="o")

    add_errorbars(ax, df["year"], df["A0_fixed_panel"],
                  df["A0_fixed_panel_lo"], df["A0_fixed_panel_hi"],
                  label="Panel (fixed b)", color="orange", marker="s")

    add_errorbars(ax, df["year"], df["A0_fixed_nonpanel"],
                  df["A0_fixed_nonpanel_lo"], df["A0_fixed_nonpanel_hi"],
                  label="Non-panel (fixed b)", color="green", marker="^")

    # 2020 >50k
    mask_2020 = df["year"] == 2020
    if not df["A0_fixed_2020_gt50k"].isna().all():
        add_errorbars(ax,
                      df["year"][mask_2020], df["A0_fixed_2020_gt50k"][mask_2020],
                      df["A0_fixed_2020_gt50k_lo"][mask_2020],
                      df["A0_fixed_2020_gt50k_hi"][mask_2020],
                      label="2020 (>50k, fixed b)", color=COL_50K, marker="D")

    ax.set_title("Baseline Area A₀ Over Time (Fixed b = 5/6)")
    ax.set_xlabel("Year")
    ax.set_ylabel("A₀ (hectares)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "A0_fixed_over_time.png"), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print(f"Loaded {CSV_FILE}")
    print("Generating plots…")

    plot_b_over_time(df)
    plot_a_over_time(df)
    plot_A0_over_time(df)

    # Fixed-b = 5/6 plots
    plot_a_fixed_over_time(df)
    plot_A0_fixed_over_time(df)

    print(f"Plots saved under: {OUTDIR}/")


if __name__ == "__main__":
    main()