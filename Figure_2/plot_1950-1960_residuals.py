import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

SQMI_TO_HA = 258.9988110336  # hectares per square mile


def load_ua_1950_1960(path):
    """
    Parse the '1950 1960-Table 1.csv'-style file into a clean DataFrame
    with columns: place, pop1960, area1960, pop1950, area1950.
    """
    # Read raw lines and split by comma
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rows.append(line.strip().split(','))

    # Pad to same length
    max_cols = max(len(r) for r in rows)
    rows_padded = [r + [''] * (max_cols - len(r)) for r in rows]
    df_raw = pd.DataFrame(rows_padded)

    # From inspection:
    # col 3: Place name
    # col 4: 1960 Pop
    # col 5: 1960 Area (sq miles)
    # col 8: 1950 Pop
    # col 9: 1950 Area (sq miles)
    place = df_raw.iloc[:, 3]
    pop60 = pd.to_numeric(df_raw.iloc[:, 4], errors="coerce")
    area60 = pd.to_numeric(df_raw.iloc[:, 5], errors="coerce")
    pop50 = pd.to_numeric(df_raw.iloc[:, 8], errors="coerce")
    area50 = pd.to_numeric(df_raw.iloc[:, 9], errors="coerce")

    clean = pd.DataFrame({
        "place": place,
        "pop1960": pop60,
        "area1960": area60,
        "pop1950": pop50,
        "area1950": area50,
    })

    # Drop rows without a real place name
    clean = clean[clean["place"].notna() & (clean["place"] != "")].copy()

    return clean


def fit_scaling(pop, area, place):
    """
    Fit log10(area) = a + b log10(pop) and return stats:
    n, a, b, CIs, R2, AIC(Normal), AIC(Student-t), df_t, A0 in hectares.
    Assumes area is in square miles; converts intercept to hectares.
    Also prints the 5 most negative and 5 most positive residuals
    with the correct place names.
    """
    # Filter valid entries
    mask = (~pop.isna()) & (~area.isna()) & (pop > 0) & (area > 0)
    pop_f = pop[mask]
    area_f = area[mask]
    place_f = place[mask]

    x = np.log10(pop_f.values)
    y = np.log10(area_f.values)  # area in sq miles

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    a, b = model.params
    ci = model.conf_int()
    try:
        a_ci = (float(ci.iloc[0, 0]), float(ci.iloc[0, 1]))
        b_ci = (float(ci.iloc[1, 0]), float(ci.iloc[1, 1]))
    except AttributeError:
        ci = np.asarray(ci)
        a_ci = (float(ci[0, 0]), float(ci[0, 1]))
        b_ci = (float(ci[1, 0]), float(ci[1, 1]))

    r2 = float(model.rsquared)
    n = int(mask.sum())
    resid = model.resid

    # Intercept in original units (sq miles), then hectares
    A0_mi2 = 10**a
    A0_mi2_ci = (10**a_ci[0], 10**a_ci[1])
    A0_ha = A0_mi2 * SQMI_TO_HA
    A0_ha_ci = (A0_mi2_ci[0] * SQMI_TO_HA,
                A0_mi2_ci[1] * SQMI_TO_HA)

    # Residuals dataframe with correct place names
    df_resid = pd.DataFrame({
        "place": place_f.values,
        "population": pop_f.values,
        "area": area_f.values,
        "residual": resid
    })

    # Identify top/bottom 5 residuals
    lowest = df_resid.nsmallest(5, "residual")
    highest = df_resid.nlargest(5, "residual")

    print("\n---- Most Negative Residuals ----")
    print(lowest.to_string(index=False))

    print("\n---- Most Positive Residuals ----")
    print(highest.to_string(index=False))

    # Normal fit
    mu_n, sig_n = stats.norm.fit(resid)
    ll_n = np.sum(stats.norm.logpdf(resid, mu_n, sig_n))
    aic_n = 2*2 - 2*ll_n

    # Student-t fit
    t_df, loc, scale = stats.t.fit(resid)
    ll_t = np.sum(stats.t.logpdf(resid, t_df, loc, scale))
    aic_t = 2*3 - 2*ll_t

    return {
        "n": n,
        "a": a,
        "b": b,
        "a_ci": a_ci,
        "b_ci": b_ci,
        "R2": r2,
        "A0_ha": A0_ha,
        "A0_ha_ci": A0_ha_ci,
        "AIC_normal": aic_n,
        "AIC_student_t": aic_t,
        "df_t": t_df,
        "resid": resid,
        "x": x,
        "y": y,
        "model": model,
    }


def plot_results(year, fit):
    x = fit["x"]
    y = fit["y"]
    model = fit["model"]
    resid = fit["resid"]

    # Scaling plot
    plt.figure(figsize=(7, 5))
    plt.scatter(10**x, 10**y, s=10, alpha=0.6)
    xx = np.linspace(x.min(), x.max(), 300)
    yy = model.params[0] + model.params[1]*xx
    plt.plot(10**xx, 10**yy, "r-", lw=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Population")
    plt.ylabel("Area (sq miles)")
#    plt.title(f"Urban Areas {year}: Area vs Population (log-log)")
    plt.tight_layout()
    plt.savefig('scaling_'+year+'.pdf')
    plt.show()

    # Residuals histogram with fits
    mu_n, sig_n = stats.norm.fit(resid)
    t_df, t_loc, t_scale = stats.t.fit(resid)
    xs = np.linspace(resid.min(), resid.max(), 400)
    pdf_n = stats.norm.pdf(xs, mu_n, sig_n)
    pdf_t = stats.t.pdf(xs, t_df, t_loc, t_scale)

    plt.figure(figsize=(7, 5))
    plt.hist(resid, bins=40, density=True, alpha=0.6)
    plt.plot(xs, pdf_n, label=f"Normal (AIC={fit['AIC_normal']:.1f})")
    plt.plot(xs, pdf_t, label=f"Student-t (AIC={fit['AIC_student_t']:.1f}, df={fit['df_t']:.1f})")
    plt.xlabel("Residual (log10 area)")
    plt.ylabel("Density")
    #plt.title(f"Residuals {year}")
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_'+year+'.pdf')
    plt.show()

    # QQ-plots
    plt.figure(figsize=(12, 5))

    # vs Normal
    plt.subplot(1, 2, 1)
    stats.probplot(resid, dist="norm", sparams=(mu_n, sig_n), plot=plt)
    plt.title(f"{year} QQ-plot vs Normal")

    # vs Student-t
    plt.subplot(1, 2, 2)
    stats.probplot(resid, dist=stats.t, sparams=(t_df, t_loc, t_scale), plot=plt)
    plt.title(f"{year} QQ-plot vs Student-t")

    plt.tight_layout()
    plt.savefig('QQ_'+year+'.pdf')
    plt.show()


def main():
    path = "1950_1960-Table_1.csv"  # adjust path if needed
    clean = load_ua_1950_1960(path)

    # 1960 fit
    fit60 = fit_scaling(clean["pop1960"], clean["area1960"], clean["place"])
    # 1950 fit (only rows with 1950 pop & area)
    fit50 = fit_scaling(clean["pop1950"], clean["area1950"], clean["place"])

    for year, fit in [("1960", fit60), ("1950", fit50)]:
        print(f"\n=== Urban Areas {year} ===")
        print(f"n = {fit['n']}")
        print(f"Slope b = {fit['b']:.4f}  "
              f"95% CI: [{fit['b_ci'][0]:.4f}, {fit['b_ci'][1]:.4f}]")
        print(f"Intercept a (log10 sq mi) = {fit['a']:.4f}  "
              f"95% CI: [{fit['a_ci'][0]:.4f}, {fit['a_ci'][1]:.4f}]")
        print(f"Intercept A0 (hectares) = {fit['A0_ha']:.2f}  "
              f"95% CI: [{fit['A0_ha_ci'][0]:.2f}, {fit['A0_ha_ci'][1]:.2f}]")
        print(f"R² = {fit['R2']:.4f}")
        print(f"AIC Normal   = {fit['AIC_normal']:.2f}")
        print(f"AIC Student-t = {fit['AIC_student_t']:.2f}  "
              f"(df ≈ {fit['df_t']:.2f})")

        # Plots (comment out if you only want numbers)
        plot_results(year, fit)


if __name__ == "__main__":
    main()