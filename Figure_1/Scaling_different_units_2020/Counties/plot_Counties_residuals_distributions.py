import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ---- configuration ----
path = "R50059151_SL050_Counties.txt"  # update path as needed
pop_col = "Total Population"
area_col = "Area Total"  # total area (land + water)

# ---- load and clean data ----
df = pd.read_csv(path, sep="\t")

# Drop repeated header row
df = df[df["FIPS"] != "Geo_geoid_"].copy()

# Convert to numeric
df[pop_col] = pd.to_numeric(df[pop_col], errors="coerce")
df[area_col] = pd.to_numeric(df[area_col], errors="coerce")

# Drop missing
df_clean = df.dropna(subset=[pop_col, area_col]).copy()

# ---- log10 transform ----
x = np.log10(df_clean[pop_col].values)   # log10 population
y = np.log10(df_clean[area_col].values)  # log10 total area

# ---- linear regression: y = a + b x ----
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

a = model.params[0]
b = model.params[1]

# 95% confidence intervals
conf_int = model.conf_int(alpha=0.05)
a_ci_low, a_ci_high = conf_int[0]
b_ci_low, b_ci_high = conf_int[1]

print("Scaling fit (log10 A = a + b log10 P):")
print(f"  a = {a:.6f}  (95% CI: [{a_ci_low:.6f}, {a_ci_high:.6f}])")
print(f"  b = {b:.6f}  (95% CI: [{b_ci_low:.6f}, {b_ci_high:.6f}])")
print()

# ---- residuals ----
residuals = model.resid

df_clean["residual"] = residuals
df_sorted = df_clean.sort_values("residual")

lowest = df_sorted.head(5)[["FIPS", "NAME", "residual"]]
highest = df_sorted.tail(5)[["FIPS", "NAME", "residual"]]

print("Lowest residuals (too small area for their population):")
print(lowest.to_string(index=False))
print()
print("Highest residuals (too large area for their population):")
print(highest.to_string(index=False))
print()

# ---- distribution fits ----
# Normal
norm_mu, norm_sigma = stats.norm.fit(residuals)
ll_norm = np.sum(stats.norm.logpdf(residuals, loc=norm_mu, scale=norm_sigma))
k_norm = 2
aic_norm = 2 * k_norm - 2 * ll_norm

# Student-t
t_df, t_loc, t_scale = stats.t.fit(residuals)
ll_t = np.sum(stats.t.logpdf(residuals, df=t_df, loc=t_loc, scale=t_scale))
k_t = 3
aic_t = 2 * k_t - 2 * ll_t

print(f"Normal fit:    mu = {norm_mu:.4f}, sigma = {norm_sigma:.4f}, AIC = {aic_norm:.1f}")
print(f"Student-t fit: df = {t_df:.4f}, loc = {t_loc:.4f}, scale = {t_scale:.4f}, AIC = {aic_t:.1f}")

# ---- plots ----

# 1. Scaling plot (log-log)
plt.figure(figsize=(7,5))
plt.scatter(df_clean[pop_col], df_clean[area_col], alpha=0.3)
xx = np.logspace(np.log10(df_clean[pop_col].min()),
                 np.log10(df_clean[pop_col].max()), 300)
yy = 10**(a + b*np.log10(xx))
plt.plot(xx, yy, 'r-', lw=2)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Total Population")
plt.ylabel("Total Area")
#plt.title("Scaling Relation: Total Area vs Population (MSAs)")
plt.tight_layout()
plt.savefig('scaling_Counties.pdf')
plt.show()

# 2. Residual histogram with Normal & Student-t fits
plt.figure()
plt.hist(residuals, bins=40, density=True, alpha=0.6)

x_grid = np.linspace(residuals.min(), residuals.max(), 400)
pdf_norm = stats.norm.pdf(x_grid, loc=norm_mu, scale=norm_sigma)
pdf_t = stats.t.pdf(x_grid, df=t_df, loc=t_loc, scale=t_scale)

plt.plot(x_grid, pdf_norm, label=f"Normal (AIC={aic_norm:.1f})")
plt.plot(x_grid, pdf_t, label=f"Student-t (AIC={aic_t:.1f})")
plt.xlabel("Residual")
plt.ylabel("Density")
plt.title("Residuals with Normal and Student-t fits")
plt.legend()
plt.savefig('distribution_Counties.pdf')

# 3. QQ plot vs Normal
plt.figure()
stats.probplot(residuals, dist="norm",
               sparams=(norm_mu, norm_sigma), plot=plt)
plt.title("QQ-plot vs Normal")

# 4. QQ plot vs Student-t
plt.figure()
stats.probplot(residuals, dist=stats.t,
               sparams=(t_df, t_loc, t_scale), plot=plt)
plt.title("QQ-plot vs Student-t")

plt.tight_layout()
plt.savefig('QQ_Counties.pdf')
plt.show()