import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, norm, t, probplot

# ------------------------------------------------------
# Load and clean data (Metropolitan Areas file)
# ------------------------------------------------------
df = pd.read_csv("R50059143_SL320_MSAs.txt", sep=None, engine="python")

df2 = df.drop(0).copy()
df2["Area Total"] = pd.to_numeric(df2["Area Total"], errors="coerce")
df2["Total Population"] = pd.to_numeric(df2["Total Population"], errors="coerce")
df2 = df2.dropna(subset=["Area Total", "Total Population"])

# ------------------------------------------------------
# Regression in log-log space:
# log10(Area) = a + b log10(Pop)
# ------------------------------------------------------
x = np.log10(df2["Total Population"])
y = np.log10(df2["Area Total"])
n = len(x)

slope, intercept, r_value, p_value, std_err_slope = linregress(x, y)
y_pred = intercept + slope * x
residuals = y - y_pred

# ------------------------------------------------------
# Compute 95% CI for slope and intercept
# ------------------------------------------------------
dfree = n - 2
from scipy.stats import t as tdist
t_crit = tdist.ppf(0.975, dfree)

# Slope CI
slope_CI = (slope - t_crit * std_err_slope,
            slope + t_crit * std_err_slope)

# Intercept CI
s_err = np.sqrt(np.sum((y - y_pred)**2) / dfree)
x_mean = np.mean(x)
Sxx = np.sum((x - x_mean)**2)
std_err_intercept = s_err * np.sqrt(1/n + x_mean**2 / Sxx)
intercept_CI = (intercept - t_crit * std_err_intercept,
                intercept + t_crit * std_err_intercept)

print("\n---- Scaling Fit ----")
print("Slope:", slope, "95% CI:", slope_CI)
print("Intercept:", intercept, "95% CI:", intercept_CI)

# ------------------------------------------------------
# Most extreme residuals
# ------------------------------------------------------
df2["residual"] = residuals
most_negative = df2.nsmallest(8, "residual")[["NAME", "residual"]]
most_positive = df2.nlargest(8, "residual")[["NAME", "residual"]]

print("\n---- Most Negative Residuals (smallest area) ----")
print(most_negative)

print("\n---- Most Positive Residuals (largest area) ----")
print(most_positive)

# ------------------------------------------------------
# Fit Normal and Student-t distributions
# ------------------------------------------------------
mu_n, sigma_n = norm.fit(residuals)
df_t, loc_t, scale_t = t.fit(residuals)

def AIC(logL, k): return 2 * k - 2 * logL
def loglik(dist, params):
    return np.sum(np.log(dist.pdf(residuals, *params)))

AIC_norm = AIC(loglik(norm, (mu_n, sigma_n)), 2)
AIC_t = AIC(loglik(t, (df_t, loc_t, scale_t)), 3)

print("\n---- AIC ----")
print("AIC Normal:   ", AIC_norm)
print("AIC Student-t:", AIC_t)
print("Student-t df: ", df_t)

# ------------------------------------------------------
# Plot 1: Scaling Plot
# ------------------------------------------------------
plt.figure(figsize=(7,5))
plt.scatter(df2["Total Population"], df2["Area Total"], alpha=0.3)
xx = np.logspace(np.log10(df2["Total Population"].min()),
                 np.log10(df2["Total Population"].max()), 300)
yy = 10**(intercept + slope*np.log10(xx))
plt.plot(xx, yy, 'r-', lw=2)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Total Population")
plt.ylabel("Total Area")
#plt.title("Scaling Relation: Total Area vs Population (MSAs)")
plt.tight_layout()
plt.savefig('scaling_MSAs.pdf')
plt.show()

# ------------------------------------------------------
# Plot 2: Residual distribution with fits
# ------------------------------------------------------
xs = np.linspace(residuals.min(), residuals.max(), 400)
pdf_norm = norm.pdf(xs, mu_n, sigma_n)
pdf_t = t.pdf(xs, df_t, loc_t, scale_t)

plt.figure(figsize=(7,5))
plt.hist(residuals, bins=40, density=True, alpha=0.5)
plt.plot(xs, pdf_norm, label="Normal", lw=2)
plt.plot(xs, pdf_t, label=f"Student-t (df={df_t:.1f})", lw=2)
plt.legend()
plt.xlabel("Residuals (log10 area)")
plt.ylabel("Density")
#plt.title("Residual Distribution (MSAs) with Normal & Student-t Fits")
plt.tight_layout()
plt.savefig('distributions_MSAs.pdf')
plt.show()

# ------------------------------------------------------
# Plot 3: QQ-plots
# ------------------------------------------------------
plt.figure(figsize=(12,5))

# QQ Normal
plt.subplot(1,2,1)
probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-plot vs Normal (MSAs)")

# QQ Student-t
plt.subplot(1,2,2)
t_dist = t(df_t, loc_t, scale_t)
theoretical_q = t_dist.ppf((np.arange(len(residuals))+0.5)/len(residuals))
plt.scatter(theoretical_q, np.sort(residuals), s=10)
minv, maxv = theoretical_q.min(), theoretical_q.max()
plt.plot([minv,maxv],[minv,maxv],'r-',lw=2)
plt.title("QQ-plot vs Student-t (MSAs)")
plt.tight_layout()
plt.savefig('QQ_MSAs.pdf')
plt.show()
