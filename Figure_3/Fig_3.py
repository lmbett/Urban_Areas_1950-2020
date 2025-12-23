import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# ============================================================
# Constants
# ============================================================

SQMI_TO_HA = 258.9988110336  # hectares per square mile


# ============================================================
# Helper functions
# ============================================================

def confint_safely(model):
    """
    Return (a_ci, b_ci) as tuples (low, high), robust to ndarray/DataFrame.
    """
    ci = model.conf_int()
    if isinstance(ci, np.ndarray):
        ci = pd.DataFrame(ci)
    a_ci = (float(ci.iloc[0, 0]), float(ci.iloc[0, 1]))
    b_ci = (float(ci.iloc[1, 0]), float(ci.iloc[1, 1]))
    return a_ci, b_ci


def fit_scaling(pop, area, min_pop=0):
    """
    Fit log10(area) = a + b log10(pop) with pop > min_pop and area > 0.
    Returns parameters, 95% CIs, R^2, sample size, and A0 in hectares.
    """
    mask = (~pop.isna()) & (~area.isna()) & (pop > min_pop) & (area > 0)
    pop_f = pop[mask]
    area_f = area[mask]

    x = np.log10(pop_f.values)
    y = np.log10(area_f.values)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    a, b = model.params
    a = float(a)
    b = float(b)

    a_ci, b_ci = confint_safely(model)
    r2 = float(model.rsquared)
    n = int(mask.sum())

    # Intercept in sq mi and hectares
    A0_sqmi = 10 ** a
    A0_ci_sqmi = (10 ** a_ci[0], 10 ** a_ci[1])
    A0_ha = A0_sqmi * SQMI_TO_HA
    A0_ha_ci = (A0_ci_sqmi[0] * SQMI_TO_HA, A0_ci_sqmi[1] * SQMI_TO_HA)

    return {
        "n": n,
        "a": a,
        "b": b,
        "a_ci": a_ci,
        "b_ci": b_ci,
        "R2": r2,
        "A0_ha": A0_ha,
        "A0_ha_ci": A0_ha_ci
    }


# ============================================================
# Loaders for each dataset (using your confirmed column mappings)
# ============================================================

def load_1950_1960(path):
    """
    1950 1960-Table 1.csv
    place = col 3
    1960: pop60 = col 4, area60 = col 5
    1950: pop50 = col 8, area50 = col 9
    """
    df_raw = pd.read_csv(path)
    place = df_raw.iloc[:, 3]
    pop60 = pd.to_numeric(df_raw.iloc[:, 4], errors="coerce")
    area60 = pd.to_numeric(df_raw.iloc[:, 5], errors="coerce")
    pop50 = pd.to_numeric(df_raw.iloc[:, 8], errors="coerce")
    area50 = pd.to_numeric(df_raw.iloc[:, 9], errors="coerce")
    return place, pop50, area50, pop60, area60


def load_1970(path):
    """
    1970-Table 1.csv
    place = Unnamed: 9
    pop70 = Unnamed: 10
    area70 = Unnamed: 11
    """
    df = pd.read_csv(path)
    place = df["Unnamed: 9"]
    pop70 = pd.to_numeric(df["Unnamed: 10"], errors="coerce")
    area70 = pd.to_numeric(df["Unnamed: 11"], errors="coerce")
    return place, pop70, area70


def load_1980(path):
    """
    1980-Table_1.csv
    place = Unnamed: 3
    pop80 = Unnamed: 4
    area80 = Unnamed: 5
    """
    df = pd.read_csv(path)
    place = df["Unnamed: 3"]
    pop80 = pd.to_numeric(df["Unnamed: 4"], errors="coerce")
    area80 = pd.to_numeric(df["Unnamed: 5"], errors="coerce")
    return place, pop80, area80


def load_1990(path):
    """
    1990-Table_1.csv
    place = Unnamed: 3
    pop90 = Unnamed: 4 (with commas)
    area90 = Unnamed: 5
    """
    df = pd.read_csv(path)
    place = df["Unnamed: 3"]
    pop90 = pd.to_numeric(df["Unnamed: 4"].astype(str).str.replace(",", ""), errors="coerce")
    area90 = pd.to_numeric(df["Unnamed: 5"], errors="coerce")
    return place, pop90, area90


def load_2000_2010(path):
    """
    2000-2010-Table_1.csv

    2010:
        place10 = Unnamed: 2  (UANAME)
        pop10   = Unnamed: 5  (UAPOP)
        area10  = Unnamed: 7  (UAAREALAND, sq mi)

    2000:
        place00 = Unnamed: 4  (UA00NAME)
        pop00   = Unnamed: 6  (UA00POP)
        area00  = Unnamed: 8  (UA00AREALAND, sq mi)
    """
    df = pd.read_csv(path)

    place10 = df["Unnamed: 2"]
    pop10 = pd.to_numeric(df["Unnamed: 5"], errors="coerce")
    area10 = pd.to_numeric(df["Unnamed: 7"], errors="coerce")

    place00 = df["Unnamed: 4"]
    pop00 = pd.to_numeric(df["Unnamed: 6"], errors="coerce")
    area00 = pd.to_numeric(df["Unnamed: 8"], errors="coerce")

    return (place00, pop00, area00), (place10, pop10, area10)


def load_2020(path):
    """
    2020-Table_1.csv (final mapping you confirmed)
    place20 = Unnamed: 1
    pop20   = Unnamed: 2 (string with commas)
    area20  = Unnamed: 4 (AREALANDSQMI)
    """
    df = pd.read_csv(path)
    place = df["Unnamed: 1"]
    pop20 = pd.to_numeric(df["Unnamed: 2"].astype(str).str.replace(",", ""), errors="coerce")
    area20 = pd.to_numeric(df["Unnamed: 4"], errors="coerce")
    return place, pop20, area20


# ============================================================
# MAIN: per-decade fits + temporal evolution plots
# ============================================================

def main():
    # Containers for results (standard = > 50k)
    years = []
    b_gt = []
    b_gt_lo = []
    b_gt_hi = []
    a_gt = []
    a_gt_lo = []
    a_gt_hi = []
    A0_gt = []
    A0_gt_lo = []
    A0_gt_hi = []

    # Also store full-sample fits for completeness
    all_results = {}

    # ---------- 1950 & 1960 ----------
    place, pop50, area50, pop60, area60 = load_1950_1960("1950_1960-Table_1.csv")

    for yr, pop, area in [(1950, pop50, area50), (1960, pop60, area60)]:
        print(f"\n==== {yr} Urban Areas ====")

        # Full sample
        res_all = fit_scaling(pop, area, min_pop=0)
        print(f"Full sample (pop > 0):")
        print(f"  n = {res_all['n']}")
        print(f"  b = {res_all['b']:.4f},  95% CI = {res_all['b_ci']}")
        print(f"  a = {res_all['a']:.4f},  95% CI = {res_all['a_ci']}")
        print(f"  R² = {res_all['R2']:.4f}")

        # >50k
        res_gt = fit_scaling(pop, area, min_pop=50000)
        print(f"\nStandard sample (pop > 50,000):")
        print(f"  n = {res_gt['n']}")
        print(f"  b = {res_gt['b']:.4f},  95% CI = {res_gt['b_ci']}")
        print(f"  a = {res_gt['a']:.4f},  95% CI = {res_gt['a_ci']}")
        print(f"  A0 (hectares) = {res_gt['A0_ha']:.2f},  95% CI = {res_gt['A0_ha_ci']}")
        print(f"  R² = {res_gt['R2']:.4f}")

        years.append(yr)
        b_gt.append(res_gt["b"])
        b_gt_lo.append(res_gt["b_ci"][0])
        b_gt_hi.append(res_gt["b_ci"][1])
        a_gt.append(res_gt["a"])
        a_gt_lo.append(res_gt["a_ci"][0])
        a_gt_hi.append(res_gt["a_ci"][1])
        A0_gt.append(res_gt["A0_ha"])
        A0_gt_lo.append(res_gt["A0_ha_ci"][0])
        A0_gt_hi.append(res_gt["A0_ha_ci"][1])

        all_results[yr] = {"all": res_all, "gt50k": res_gt}

    # ---------- 1970 ----------
    place70, pop70, area70 = load_1970("1970-Table_1.csv")
    for yr, pop, area in [(1970, pop70, area70)]:
        print(f"\n==== {yr} Urban Areas ====")

        res_all = fit_scaling(pop, area, min_pop=0)
        print(f"Full sample (pop > 0):")
        print(f"  n = {res_all['n']}")
        print(f"  b = {res_all['b']:.4f},  95% CI = {res_all['b_ci']}")
        print(f"  a = {res_all['a']:.4f},  95% CI = {res_all['a_ci']}")
        print(f"  R² = {res_all['R2']:.4f}")

        res_gt = fit_scaling(pop, area, min_pop=50000)
        print(f"\nStandard sample (pop > 50,000):")
        print(f"  n = {res_gt['n']}")
        print(f"  b = {res_gt['b']:.4f},  95% CI = {res_gt['b_ci']}")
        print(f"  a = {res_gt['a']:.4f},  95% CI = {res_gt['a_ci']}")
        print(f"  A0 (hectares) = {res_gt['A0_ha']:.2f},  95% CI = {res_gt['A0_ha_ci']}")
        print(f"  R² = {res_gt['R2']:.4f}")

        years.append(yr)
        b_gt.append(res_gt["b"])
        b_gt_lo.append(res_gt["b_ci"][0])
        b_gt_hi.append(res_gt["b_ci"][1])
        a_gt.append(res_gt["a"])
        a_gt_lo.append(res_gt["a_ci"][0])
        a_gt_hi.append(res_gt["a_ci"][1])
        A0_gt.append(res_gt["A0_ha"])
        A0_gt_lo.append(res_gt["A0_ha_ci"][0])
        A0_gt_hi.append(res_gt["A0_ha_ci"][1])

        all_results[yr] = {"all": res_all, "gt50k": res_gt}

    # ---------- 1980 ----------
    place80, pop80, area80 = load_1980("1980-Table_1.csv")
    for yr, pop, area in [(1980, pop80, area80)]:
        print(f"\n==== {yr} Urban Areas ====")

        res_all = fit_scaling(pop, area, min_pop=0)
        print(f"Full sample (pop > 0):")
        print(f"  n = {res_all['n']}")
        print(f"  b = {res_all['b']:.4f},  95% CI = {res_all['b_ci']}")
        print(f"  a = {res_all['a']:.4f},  95% CI = {res_all['a_ci']}")
        print(f"  R² = {res_all['R2']:.4f}")

        res_gt = fit_scaling(pop, area, min_pop=50000)
        print(f"\nStandard sample (pop > 50,000):")
        print(f"  n = {res_gt['n']}")
        print(f"  b = {res_gt['b']:.4f},  95% CI = {res_gt['b_ci']}")
        print(f"  a = {res_gt['a']:.4f},  95% CI = {res_gt['a_ci']}")
        print(f"  A0 (hectares) = {res_gt['A0_ha']:.2f},  95% CI = {res_gt['A0_ha_ci']}")
        print(f"  R² = {res_gt['R2']:.4f}")

        years.append(yr)
        b_gt.append(res_gt["b"])
        b_gt_lo.append(res_gt["b_ci"][0])
        b_gt_hi.append(res_gt["b_ci"][1])
        a_gt.append(res_gt["a"])
        a_gt_lo.append(res_gt["a_ci"][0])
        a_gt_hi.append(res_gt["a_ci"][1])
        A0_gt.append(res_gt["A0_ha"])
        A0_gt_lo.append(res_gt["A0_ha_ci"][0])
        A0_gt_hi.append(res_gt["A0_ha_ci"][1])

        all_results[yr] = {"all": res_all, "gt50k": res_gt}

    # ---------- 1990 ----------
    place90, pop90, area90 = load_1990("1990-Table_1.csv")
    for yr, pop, area in [(1990, pop90, area90)]:
        print(f"\n==== {yr} Urban Areas ====")

        res_all = fit_scaling(pop, area, min_pop=0)
        print(f"Full sample (pop > 0):")
        print(f"  n = {res_all['n']}")
        print(f"  b = {res_all['b']:.4f},  95% CI = {res_all['b_ci']}")
        print(f"  a = {res_all['a']:.4f},  95% CI = {res_all['a_ci']}")
        print(f"  R² = {res_all['R2']:.4f}")

        res_gt = fit_scaling(pop, area, min_pop=50000)
        print(f"\nStandard sample (pop > 50,000):")
        print(f"  n = {res_gt['n']}")
        print(f"  b = {res_gt['b']:.4f},  95% CI = {res_gt['b_ci']}")
        print(f"  a = {res_gt['a']:.4f},  95% CI = {res_gt['a_ci']}")
        print(f"  A0 (hectares) = {res_gt['A0_ha']:.2f},  95% CI = {res_gt['A0_ha_ci']}")
        print(f"  R² = {res_gt['R2']:.4f}")

        years.append(yr)
        b_gt.append(res_gt["b"])
        b_gt_lo.append(res_gt["b_ci"][0])
        b_gt_hi.append(res_gt["b_ci"][1])
        a_gt.append(res_gt["a"])
        a_gt_lo.append(res_gt["a_ci"][0])
        a_gt_hi.append(res_gt["a_ci"][1])
        A0_gt.append(res_gt["A0_ha"])
        A0_gt_lo.append(res_gt["A0_ha_ci"][0])
        A0_gt_hi.append(res_gt["A0_ha_ci"][1])

        all_results[yr] = {"all": res_all, "gt50k": res_gt}

    # ---------- 2000 & 2010 ----------
    (place00, pop00, area00), (place10, pop10, area10) = load_2000_2010("2000-2010-Table_1.csv")
    for yr, pop, area in [(2000, pop00, area00), (2010, pop10, area10)]:
        print(f"\n==== {yr} Urban Areas ====")

        res_all = fit_scaling(pop, area, min_pop=0)
        print(f"Full sample (pop > 0):")
        print(f"  n = {res_all['n']}")
        print(f"  b = {res_all['b']:.4f},  95% CI = {res_all['b_ci']}")
        print(f"  a = {res_all['a']:.4f},  95% CI = {res_all['a_ci']}")
        print(f"  R² = {res_all['R2']:.4f}")

        res_gt = fit_scaling(pop, area, min_pop=50000)
        print(f"\nStandard sample (pop > 50,000):")
        print(f"  n = {res_gt['n']}")
        print(f"  b = {res_gt['b']:.4f},  95% CI = {res_gt['b_ci']}")
        print(f"  a = {res_gt['a']:.4f},  95% CI = {res_gt['a_ci']}")
        print(f"  A0 (hectares) = {res_gt['A0_ha']:.2f},  95% CI = {res_gt['A0_ha_ci']}")
        print(f"  R² = {res_gt['R2']:.4f}")

        years.append(yr)
        b_gt.append(res_gt["b"])
        b_gt_lo.append(res_gt["b_ci"][0])
        b_gt_hi.append(res_gt["b_ci"][1])
        a_gt.append(res_gt["a"])
        a_gt_lo.append(res_gt["a_ci"][0])
        a_gt_hi.append(res_gt["a_ci"][1])
        A0_gt.append(res_gt["A0_ha"])
        A0_gt_lo.append(res_gt["A0_ha_ci"][0])
        A0_gt_hi.append(res_gt["A0_ha_ci"][1])

        all_results[yr] = {"all": res_all, "gt50k": res_gt}

    # ---------- 2020 ----------
    place20, pop20, area20 = load_2020("2020-Table_1.csv")
    for yr, pop, area in [(2020, pop20, area20)]:
        print(f"\n==== {yr} Urban Areas ====")

        res_all = fit_scaling(pop, area, min_pop=0)
        print(f"Full sample (pop > 0):")
        print(f"  n = {res_all['n']}")
        print(f"  b = {res_all['b']:.4f},  95% CI = {res_all['b_ci']}")
        print(f"  a = {res_all['a']:.4f},  95% CI = {res_all['a_ci']}")
        print(f"  R² = {res_all['R2']:.4f}")

        res_gt = fit_scaling(pop, area, min_pop=50000)
        print(f"\nStandard sample (pop > 50,000):")
        print(f"  n = {res_gt['n']}")
        print(f"  b = {res_gt['b']:.4f},  95% CI = {res_gt['b_ci']}")
        print(f"  a = {res_gt['a']:.4f},  95% CI = {res_gt['a_ci']}")
        print(f"  A0 (hectares) = {res_gt['A0_ha']:.2f},  95% CI = {res_gt['A0_ha_ci']}")
        print(f"  R² = {res_gt['R2']:.4f}")

        years.append(yr)
        b_gt.append(res_gt["b"])
        b_gt_lo.append(res_gt["b_ci"][0])
        b_gt_hi.append(res_gt["b_ci"][1])
        a_gt.append(res_gt["a"])
        a_gt_lo.append(res_gt["a_ci"][0])
        a_gt_hi.append(res_gt["a_ci"][1])
        A0_gt.append(res_gt["A0_ha"])
        A0_gt_lo.append(res_gt["A0_ha_ci"][0])
        A0_gt_hi.append(res_gt["A0_ha_ci"][1])

        all_results[yr] = {"all": res_all, "gt50k": res_gt}

    # ========================================================
    # Build parameter evolution plots for standard sample (>50k)
    # ========================================================
    years_arr = np.array(years)
    order = np.argsort(years_arr)
    years_arr = years_arr[order]

    b_gt_arr = np.array(b_gt)[order]
    b_lo_arr = np.array(b_gt_lo)[order]
    b_hi_arr = np.array(b_gt_hi)[order]

    a_gt_arr = np.array(a_gt)[order]
    a_lo_arr = np.array(a_gt_lo)[order]
    a_hi_arr = np.array(a_gt_hi)[order]

    A0_arr = np.array(A0_gt)[order]
    A0_lo_arr = np.array(A0_gt_lo)[order]
    A0_hi_arr = np.array(A0_gt_hi)[order]

    # 1) b(t)
    plt.figure(figsize=(7, 5))
    yerr = np.vstack([b_gt_arr - b_lo_arr, b_hi_arr - b_gt_arr])
    plt.errorbar(years_arr, b_gt_arr, yerr=yerr, fmt="o", capsize=4,alpha=0.9)
    plt.plot(years_arr,b_gt_arr,'r-',lw=2,alpha=0.7)
    
    plt.axhline(y=5/6, color="grey", linewidth=3, alpha=0.6)

    plt.xlabel("Year")
    plt.ylabel("Scaling exponent")
    #plt.title("Temporal evolution of scaling exponent b (Area vs Population, land only, pop > 50k)")
    #plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('scaling_exponent_over_time.pdf')
    plt.show()

    # 2) a(t)
    plt.figure(figsize=(7, 5))
    yerr = np.vstack([a_gt_arr - a_lo_arr, a_hi_arr - a_gt_arr])
    plt.errorbar(years_arr, a_gt_arr, yerr=yerr, fmt="o", capsize=4,alpha=0.9)
    plt.plot(years_arr,a_gt_arr,'r-',lw=2,alpha=0.7)
    plt.xlabel("Year")
    plt.ylabel("Intercept a (log10 area in square miles)")
    #plt.title("Temporal evolution of intercept a (Area vs Population, land only, pop > 50k)")
    #plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('log_prefactor_over_time.pdf')
    plt.show()

    # 3) A0(t) in hectares
    plt.figure(figsize=(7, 5))
    yerr = np.vstack([A0_arr - A0_lo_arr, A0_hi_arr - A0_arr])
    plt.errorbar(years_arr, A0_arr, yerr=yerr, fmt="o", capsize=4,alpha=0.9)
    plt.plot(years_arr,A0_arr,'r-',lw=2,alpha=0.7)
    plt.xlabel("Year")
    plt.ylabel("A0 (hectares, A(N) at N = 1)")
    #plt.title("Temporal evolution of intercept A0 in hectares (Area vs Population, land only, pop > 50k)")
    #plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prefactor_hectares_over_time.pdf')
    plt.show()


if __name__ == "__main__":
    main()

