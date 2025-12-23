import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ---- config ----
path = "R50059153_SL871_ZIPCODE.txt"
pop_col = "Total Population"
area_col = "Area Total"

# ---- load & clean ----
df = pd.read_csv(path, sep="\t")

# Drop repeated header row
df = df[df["FIPS"] != "Geo_geoid_"].copy()

# Convert to numeric
df[pop_col] = pd.to_numeric(df[pop_col], errors="coerce")
df[area_col] = pd.to_numeric(df[area_col], errors="coerce")

# Exclude zeros/negatives and NaNs
df_clean = df.dropna(subset=[pop_col, area_col]).copy()
df_clean = df_clean[(df_clean[pop_col] > 0) & (df_clean[area_col] > 0)].copy()

# ---- log10 transform ----
x = np.log10(df_clean[pop_col].values)
y = np.log10(df_clean[area_col].values)

# ---- linear regression y = a + b x ----
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
a, b = model.params

conf = model.conf_int(0.05)
a_ci_low, a_ci_high = conf[0]
b_ci_low, b_ci_high = conf[1]

print("Scaling fit (log10 A = a + b log10 P):")
print(f"  a = {a:.6f}  (95% CI: [{a_ci_low:.6f}, {a_ci_high:.6f}])")
print(f"  b = {b:.6f}  (95% CI: [{b_ci_low:.6f}, {b_ci_high:.6f}])")
print()

# ---- residuals ----
residuals = model.resid
df_clean["residual"] = residuals

# Extreme residuals
lowest = df_clean.nsmallest(5, "residual")[["FIPS", "NAME", "residual"]]
highest = df_clean.nlargest(5, "residual")[["FIPS", "NAME", "residual"]]

print("Lowest residuals (smallest area for their population):")
print(lowest.to_string(index=False))
print()
print("Highest residuals (largest area for their population):")
print(highest.to_string(index=False))
print()

# ---- distribution fits ----
# Normal
norm_mu, norm_sigma = stats.norm.fit(residuals)
ll_norm = np.sum(stats.norm.logpdf(residuals, loc=norm_mu, scale=norm_sigma))
aic_norm = 2*2 - 2*ll_norm

# Student-t
t_df, t_loc, t_scale = stats.t.fit(residuals)
ll_t = np.sum(stats.t.logpdf(residuals, df=t_df, loc=t_loc, scale=t_scale))
aic_t = 2*3 - 2*ll_t

print(f"Normal fit:    mu = {norm_mu:.4f}, sigma = {norm_sigma:.4f}, AIC = {aic_norm:.1f}")
print(f"Student-t fit: df = {t_df:.4f}, loc = {t_loc:.4f}, scale = {t_scale:.4f}, AIC = {aic_t:.1f}")

# ---- plots ----

# Scaling plot
plt.figure(figsize=(7,5))
plt.scatter(df_clean[pop_col], df_clean[area_col], alpha=0.3)
xx = np.logspace(np.log10(df_clean[pop_col].min()),
                 np.log10(df_clean[pop_col].max()), 300)
yy = 10**(a + b*np.log10(xx))
plt.plot(xx, yy, 'r-', lw=2)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Total Population")
plt.ylabel("Total Area")
plt.tight_layout()
plt.savefig('scaling_ZIPs.pdf')
plt.show()


#plt.figure(figsize=(7,5))
#plt.scatter(x, y, s=5, alpha=0.3, color="orange")
#x_line = np.linspace(x.min(), x.max(), 300)
#y_line = a + b * x_line
#plt.plot(x_line, y_line, "r-", lw=2)
#plt.xlabel("log10(Total Population)")
#plt.ylabel("log10(Total Area)")
#plt.title("ZIP Codes: Total Area vs Population (log-log)")
#plt.tight_layout()
#plt.savefig('scaling_ZIP.pdf')
#plt.show()

# Residual histogram with fits
plt.figure(figsize=(7,5))
plt.hist(residuals, bins=40, density=True, alpha=0.6, color="goldenrod")
xs = np.linspace(residuals.min(), residuals.max(), 400)
pdf_norm = stats.norm.pdf(xs, loc=norm_mu, scale=norm_sigma)
pdf_t = stats.t.pdf(xs, df=t_df, loc=t_loc, scale=t_scale)
plt.plot(xs, pdf_norm, label=f"Normal (AIC={aic_norm:.1f})")
plt.plot(xs, pdf_t, label=f"Student-t (AIC={aic_t:.1f})")
plt.xlabel("Residual")
plt.ylabel("Density")
plt.title("ZIP Codes: Residuals with Normal & Student-t fits")
plt.legend()
plt.tight_layout()
plt.savefig('distribution_ZIP.pdf')
plt.show()

# QQ vs Normal
plt.figure(figsize=(6,5))
stats.probplot(residuals, dist="norm",
               sparams=(norm_mu, norm_sigma), plot=plt)
plt.title("ZIP Codes: QQ-plot vs Normal")
plt.tight_layout()
plt.savefig('QQ_ZIP_Normal.pdf')

plt.show()

# QQ vs Student-t
plt.figure(figsize=(6,5))
stats.probplot(residuals, dist=stats.t,
               sparams=(t_df, t_loc, t_scale), plot=plt)
plt.title("ZIP Codes: QQ-plot vs Student-t")
plt.tight_layout()
plt.savefig('QQ_ZIP_Student-t.pdf')
plt.show()
