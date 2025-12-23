#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, probplot
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------
# Load and clean data
# --------------------------------------------------

fname = "R50059137_SL159_places.txt"
df = pd.read_csv(fname, sep="\t", dtype=str)

# Replace "." with NaN
df = df.replace(".", np.nan)

# Convert to numeric
df["population"] = pd.to_numeric(df["Total Population"], errors="coerce")
df["area_land"] = pd.to_numeric(df["AREALAND"], errors="coerce")
df["name"] = df["NAME"]

# Remove zeros and missing
df = df[(df["population"] > 0) & (df["area_land"] > 0)].copy()

# Logs
df["logP"] = np.log10(df["population"])
df["logA"] = np.log10(df["area_land"])

# Drop non-finite
df = df[np.isfinite(df["logP"]) & np.isfinite(df["logA"])].copy()

print(f"Loaded {len(df)} clean Places rows.")


# --------------------------------------------------
# Scaling regression
# --------------------------------------------------
X = sm.add_constant(df["logP"])
model = sm.OLS(df["logA"], X).fit()
a, b = model.params
a_ci = model.conf_int().loc["const"].tolist()
b_ci = model.conf_int().loc["logP"].tolist()
R2 = model.rsquared
# Compute A0 (hectares) and its CI
A0 = 10**a / 10000        # convert m² → hectares
A0_lo = 10**a_ci[0] / 10000
A0_hi = 10**a_ci[1] / 10000

df["pred_logA"] = model.predict(X)
df["residual"] = df["logA"] - df["pred_logA"]


# Clean residuals
residuals = df["residual"].astype(float).values
residuals = residuals[np.isfinite(residuals)]

if len(residuals) == 0:
    raise ValueError("Residuals are empty after cleaning.")


# --------------------------------------------------
# Distribution fits
# --------------------------------------------------
mu_n, sigma_n = norm.fit(residuals)
df_t, mu_t, sigma_t = t.fit(residuals)

def AIC_normal(res):
    ll = np.sum(norm.logpdf(res, mu_n, sigma_n))
    return 2*2 - 2*ll   # 2 parameters: mean, sd

def AIC_t(res):
    ll = np.sum(t.logpdf(res, df_t, mu_t, sigma_t))
    return 2*3 - 2*ll   # 3 parameters: df, loc, scale

aic_n = AIC_normal(residuals)
aic_t = AIC_t(residuals)

print("\n=== Scaling results ===")
print(f"a = {a:.3f}  [{a_ci[0]:.3f}, {a_ci[1]:.3f}]")
print(f"b = {b:.3f}  [{b_ci[0]:.3f}, {b_ci[1]:.3f}]")
print(f"A₀ = {A0:.2f} ha  [{A0_lo:.2f}, {A0_hi:.2f}]")
print(f"R² = {R2:.4f}")

print("\nAIC Normal   =", round(aic_n, 2))
print("AIC Student-t =", round(aic_t, 2), f"(df={df_t:.2f})")


# --------------------------------------------------
# Extreme residuals
# --------------------------------------------------
print("\nMost negative residuals:")
print(df.nsmallest(10, "residual")[["name", "residual"]])

print("\nMost positive residuals:")
print(df.nlargest(10, "residual")[["name", "residual"]])


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

# 1. Scaling plot
plt.figure(figsize=(7,5))
plt.scatter(df["population"], df["area_land"], alpha=0.3)
xx = np.logspace(np.log10(df["population"].min()),
                 np.log10(df["population"].max()), 300)
yy = 10**(a + b*np.log10(xx))
plt.plot(xx, yy, 'r-', lw=2)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Total Population")
plt.ylabel("Total Area")
#plt.title("Scaling Relation: Total Area vs Population (MSAs)")
plt.tight_layout()
plt.savefig('scaling_Places.pdf')
plt.show()



# 2. Histogram with PDFs
plt.figure(figsize=(7,6))
xs = np.linspace(residuals.min(), residuals.max(), 400)

plt.hist(residuals, bins=40, density=True, alpha=0.5, label="Residuals")
plt.plot(xs, norm.pdf(xs, mu_n, sigma_n),
         label=f"Normal fit (μ={mu_n:.2f}, σ={sigma_n:.2f})")
plt.plot(xs, t.pdf(xs, df_t, mu_t, sigma_t),
         label=f"Student-t fit (df={df_t:.1f})")

plt.xlabel("Residual")
plt.ylabel("Density")
plt.title("Residual Distribution - Places")
plt.legend()
plt.tight_layout()
plt.savefig("Places_residuals_distribution.pdf", dpi=200)
plt.close()


# 3. QQ plot Normal
plt.figure(figsize=(6,6))
probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot – Normal")
plt.tight_layout()
plt.savefig("Places_QQ_Normal.pdf", dpi=200)
plt.close()


# 4. QQ plot Student-t
plt.figure(figsize=(6,6))
probplot(residuals, dist=t(df_t, mu_t, sigma_t), plot=plt)
plt.title("QQ Plot – Student-t")
plt.tight_layout()
plt.savefig("Places_QQ_t.pdf", dpi=200)
plt.close()

print("\nSaved plots:")
print("  Places_scaling_plot.pdf")
print("  Places_residuals_distribution.pdf")
print("  Places_QQ_Normal.pdf")
print("  Places_QQ_t.pdf")

