import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ------------------------------------------------------
# Load and clean data
# ------------------------------------------------------
df = pd.read_csv("R50059143_SL320_MSAs.txt", sep=None, engine="python")

# Drop repeated header row
df = df[df["CBSAFP"] != "CBSAFP"].copy() if "CBSAFP" in df.columns else df.drop(0).copy()

# Convert numeric
df["Area Total"] = pd.to_numeric(df["Area Total"], errors="coerce")
df["Total Population"] = pd.to_numeric(df["Total Population"], errors="coerce")

# ------------------------------------------------------
# Filter to METRO AREAS ONLY
# NAME field ends with "... Metro Area"
# ------------------------------------------------------
df_metro = df[df["NAME"].str.contains("Metro Area", case=False, na=False)].copy()

# Drop NAs
df_metro = df_metro.dropna(subset=["Area Total", "Total Population"])

# ------------------------------------------------------
# Regression in log-log space
# log10(Area) = a + b log10(Pop)
# ------------------------------------------------------
x = np.log10(df_metro["Total Population"].values)
y = np.log10(df_metro["Area Total"].values)

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

a, b = model.params
conf = model.conf_int(alpha=0.05)
a_ci = tuple(conf[0])
b_ci = tuple(conf[1])

print("\n---- METRO AREAS: Scaling Fit ----")
print(f"Slope b = {b:.6f}  (95% CI: [{b_ci[0]:.6f}, {b_ci[1]:.6f}])")
print(f"Intercept a = {a:.6f}  (95% CI: [{a_ci[0]:.6f}, {a_ci[1]:.6f}])")

# Residuals
residuals = model.resid
df_metro["residual"] = residuals

# ------------------------------------------------------
# Fit Normal and Student-t to residuals
# ------------------------------------------------------
norm_mu, norm_sigma = stats.norm.fit(residuals)
ll_norm = np.sum(stats.norm.logpdf(residuals, loc=norm_mu, scale=norm_sigma))
aic_norm = 2*2 - 2*ll_norm

t_df, t_loc, t_scale = stats.t.fit(residuals)
ll_t = np.sum(stats.t.logpdf(residuals, df=t_df, loc=t_loc, scale=t_scale))
aic_t = 2*3 - 2*ll_t

print("\n---- METRO AREAS: AIC ----")
print(f"Normal AIC:    {aic_norm:.2f}")
print(f"Student-t AIC: {aic_t:.2f}")
print(f"Student-t df:  {t_df:.2f}")

# ------------------------------------------------------
# Extreme residuals
# ------------------------------------------------------
lowest = df_metro.nsmallest(5, "residual")[["CBSAFP","NAME","residual"]] if "CBSAFP" in df_metro else df_metro.nsmallest(5, "residual")[["NAME","residual"]]
highest = df_metro.nlargest(5, "residual")[["CBSAFP","NAME","residual"]] if "CBSAFP" in df_metro else df_metro.nlargest(5, "residual")[["NAME","residual"]]

print("\n---- Most Negative Residuals ----")
print(lowest.to_string(index=False))

print("\n---- Most Positive Residuals ----")
print(highest.to_string(index=False))

# ------------------------------------------------------
# Plot 1: Scaling Plot
# ------------------------------------------------------
plt.figure(figsize=(7,5))
plt.scatter(df_metro["Total Population"], df_metro["Area Total"], alpha=0.3)
xx = np.logspace(np.log10(df_metro["Total Population"].min()),
                 np.log10(df_metro["Total Population"].max()), 300)
yy = 10**(a + b*np.log10(xx))
plt.plot(xx, yy, 'r-', lw=2)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Total Population")
plt.ylabel("Total Area")
#plt.title("Scaling Relation: Total Area vs Population (MSAs)")
plt.tight_layout()
plt.savefig('scaling_MSAs_only.pdf')
plt.show()

#plt.scatter(x, y, alpha=0.5)
#xx = np.linspace(x.min(), x.max(), 300)
#yy = a + b * xx
#plt.plot(xx, yy, 'r-', lw=2)
#plt.xscale('log'); plt.yscale('log')
#plt.xlabel("Total Population (log10)")
#plt.ylabel("Total Area (log10)")
#plt.title("Metro Areas: Total Area vs Population (log-log)")
#plt.tight_layout()
#plt.savefig('scaling_MSAs_only.pdf')
#plt.show()

# ------------------------------------------------------
# Plot 2: Residual Distribution
# ------------------------------------------------------
xs = np.linspace(residuals.min(), residuals.max(), 400)
pdf_norm = stats.norm.pdf(xs, norm_mu, norm_sigma)
pdf_t = stats.t.pdf(xs, df=t_df, loc=t_loc, scale=t_scale)

plt.figure(figsize=(7,5))
plt.hist(residuals, bins=40, density=True, alpha=0.6)
plt.plot(xs, pdf_norm, label=f"Normal (AIC={aic_norm:.1f})")
plt.plot(xs, pdf_t, label=f"Student-t (AIC={aic_t:.1f})")
plt.legend()
plt.xlabel("Residual (log10 area)")
plt.ylabel("Density")
plt.title("Metro Areas: Residual Distribution")
plt.tight_layout()
plt.savefig('distributions_MSAs_only.pdf')
plt.show()

# ------------------------------------------------------
# Plot 3: QQ-plots
# ------------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
stats.probplot(residuals, dist="norm", sparams=(norm_mu,norm_sigma), plot=plt)
plt.title("QQ-plot vs Normal (Metro Areas)")

plt.subplot(1,2,2)
stats.probplot(residuals, dist=stats.t, sparams=(t_df,t_loc,t_scale), plot=plt)
plt.title("QQ-plot vs Student-t (Metro Areas)")
plt.tight_layout()
plt.savefig('QQ_MSAs_only.pdf')
plt.show()