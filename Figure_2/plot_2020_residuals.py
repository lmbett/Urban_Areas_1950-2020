import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# Load 2020 file
df = pd.read_csv("2020-Table_1.csv")

# Relevant fields:
# Unnamed: 2 -> NAME (UA name)
# Unnamed: 3 -> POP (string)
# Unnamed: 5 -> AREALANDSQMI (area in sq miles)

place20 = df["Unnamed: 1"]
pop20   = pd.to_numeric(df["Unnamed: 2"].str.replace(",", ""), errors="coerce")
area20  = pd.to_numeric(df["Unnamed: 4"], errors="coerce")


# ======================================================
# Scaling + residual analysis + plots
# ======================================================
def scaling_analysis(place, pop, area, label="YEAR"):

    mask = (
        place.notna() &
        pop.notna() &
        area.notna() &
        (pop > 0) &
        (area > 0)
    )

    place_f = place[mask]
    pop_f   = pop[mask]
    area_f  = area[mask]

    x = np.log10(pop_f.values)
    y = np.log10(area_f.values)

    # Fit scaling
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # CI robust
    ci = model.conf_int()
    if isinstance(ci, np.ndarray):
        ci = pd.DataFrame(ci)

    a = float(model.params[0])
    b = float(model.params[1])
    a_ci = (float(ci.iloc[0,0]), float(ci.iloc[0,1]))
    b_ci = (float(ci.iloc[1,0]), float(ci.iloc[1,1]))
    r2 = float(model.rsquared)
    n = len(x)

    print(f"\n===== {label} Scaling Results =====")
    print(f"n = {n}")
    print(f"Slope b = {b:.4f}   95% CI: [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")
    print(f"Intercept a = {a:.4f}   95% CI: [{a_ci[0]:.4f}, {a_ci[1]:.4f}]")
    print(f"R² = {r2:.4f}")

    # Residuals
    resid = model.resid
    resdf = pd.DataFrame({
        "place": place_f.values,
        "pop":   pop_f.values,
        "area":  area_f.values,
        "resid": resid
    })

    lowest  = resdf.nsmallest(5, "resid")
    highest = resdf.nlargest(5, "resid")

    print(f"\n===== {label} MOST NEGATIVE RESIDUALS =====")
    print(lowest.to_string(index=False))

    print(f"\n===== {label} MOST POSITIVE RESIDUALS =====")
    print(highest.to_string(index=False))

    # Fits: Normal vs Student-t
    mu_n, sig_n = stats.norm.fit(resid)
    ll_n = np.sum(stats.norm.logpdf(resid, mu_n, sig_n))
    aic_normal = 2*2 - 2*ll_n

    t_df, t_loc, t_scale = stats.t.fit(resid)
    ll_t = np.sum(stats.t.logpdf(resid, t_df, t_loc, t_scale))
    aic_t = 2*3 - 2*ll_t

    print(f"\n--- AIC Comparison ({label}) ---")
    print(f"AIC (Normal)    = {aic_normal:.2f}")
    print(f"AIC (Student-t) = {aic_t:.2f}   df = {t_df:.2f}")

    # ======================================================
    # PLOTS
    # ======================================================

    # Scaling plot
    plt.figure(figsize=(7,5))
    plt.scatter(pop_f, area_f, s=10, alpha=0.6)
    xx = np.linspace(x.min(), x.max(), 300)
    yy = model.params[0] + model.params[1] * xx
    plt.plot(10**xx, 10**yy, 'r-', lw=2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"Population {label}")
    plt.ylabel("Area (sq miles)")
    #plt.title(f"Urban Areas {label}: Area vs Population (log-log)")
    plt.tight_layout()
    plt.savefig('scaling_2020.pdf')
    plt.show()

    # Residual histogram + PDFs
    xs = np.linspace(resid.min(), resid.max(), 400)
    pdf_norm = stats.norm.pdf(xs, mu_n, sig_n)
    pdf_t = stats.t.pdf(xs, t_df, t_loc, t_scale)

    plt.figure(figsize=(7,5))
    plt.hist(resid, bins=40, density=True, alpha=0.6)
    plt.plot(xs, pdf_norm, label=f"Normal (AIC={aic_normal:.1f})")
    plt.plot(xs, pdf_t,   label=f"Student-t (AIC={aic_t:.1f}, df={t_df:.1f})")
    plt.xlabel("Residual (log10 area)")
    plt.ylabel("Density")
    #plt.title(f"{label} Residual Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_2020.pdf')
    plt.show()

    # QQ plots
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    stats.probplot(resid, dist="norm", sparams=(mu_n, sig_n), plot=plt)
    plt.title(f"{label} QQ-Plot vs Normal")

    plt.subplot(1,2,2)
    stats.probplot(resid, dist=stats.t, sparams=(t_df, t_loc, t_scale), plot=plt)
    plt.title(f"{label} QQ-Plot vs Student-t")

    plt.tight_layout()
    plt.savefig('QQ_2020.pdf')
    plt.show()

    # Return full results for tables
    return {
        "n": n, "a": a, "b": b,
        "a_ci": a_ci, "b_ci": b_ci,
        "R2": r2,
        "AIC_normal": aic_normal,
        "AIC_student_t": aic_t,
        "df_t": t_df,
        "lowest": lowest,
        "highest": highest
    }


# ======================================================
# RUN ANALYSIS
# ======================================================
results_2020 = scaling_analysis(place20, pop20, area20, label="2020")