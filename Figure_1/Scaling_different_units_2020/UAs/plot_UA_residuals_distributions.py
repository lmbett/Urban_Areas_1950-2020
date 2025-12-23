import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, norm, t, probplot

# ------------------------------------------------------
# Load and clean data
# ------------------------------------------------------
df = pd.read_csv("R50059156_SL420_UAs.txt", sep=None, engine="python")

df2 = df.drop(0).copy()
df2["Area Total"] = pd.to_numeric(df2["Area Total"], errors="coerce")
df2["Total Population"] = pd.to_numeric(df2["Total Population"], errors="coerce")
df2 = df2.dropna(subset=["Area Total", "Total Population"])

# ------------------------------------------------------
# Regression: log10(Area) = a + b * log10(Pop)
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
# Fit Normal and Student-t distributions
# ------------------------------------------------------
# Normal
mu_n, sigma_n = norm.fit(residuals)

# Student-t
df_t, loc_t, scale_t = t.fit(residuals)

# AIC computations
def AIC(logL, k):
    return 2*k - 2*logL

def loglikelihood(dist, params):
    return np.sum(np.log(dist.pdf(residuals, *params)))

LL_norm = loglikelihood(norm, (mu_n, sigma_n))
LL_t = loglikelihood(t, (df_t, loc_t, scale_t))

AIC_norm = AIC(LL_norm, 2)
AIC_t = AIC(LL_t, 3)

print("AIC Normal:    ", AIC_norm)
print("AIC Student-t: ", AIC_t)
print("Student-t degrees of freedom:", df_t)

# ------------------------------------------------------
# Plot 1: Scaling plot
# ------------------------------------------------------
plt.figure(figsize=(7,5))
plt.scatter(df2["Total Population"], df2["Area Total"], alpha=0.2, label="Urban Areas")
xx = np.logspace(np.log10(df2["Total Population"].min()),
                 np.log10(df2["Total Population"].max()), 300)
yy = 10**(intercept + slope*np.log10(xx))
plt.plot(xx, yy, color="red", label=f"Fit slope={slope:.3f}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Total Population")
plt.ylabel("Total Area")
#plt.title("Scaling Relation: Total Area vs Population")
#plt.legend()
plt.tight_layout()
plt.savefig('scaling_UA.pdf')
plt.show()

# ------------------------------------------------------
# Plot 2: Residual distribution with Normal + Student-t fits
# ------------------------------------------------------
xs = np.linspace(residuals.min(), residuals.max(), 400)
pdf_norm = norm.pdf(xs, mu_n, sigma_n)
pdf_t = t.pdf(xs, df_t, loc_t, scale_t)

plt.figure(figsize=(7,5))
plt.hist(residuals, bins=40, density=True, alpha=0.5, label="Residuals")
plt.plot(xs, pdf_norm, label="Normal fit", lw=2)
plt.plot(xs, pdf_t, label=f"Student-t fit (df={df_t:.1f})", lw=2)
plt.xlabel("Residuals (log10 area)")
plt.ylabel("Density")
plt.title("Residual Distribution with Normal & Student-t Fits")
plt.legend()
plt.tight_layout()
plt.savefig('distributions_UA.pdf')
plt.show()

# ------------------------------------------------------
# Plot 3: QQ-plots for Normal and Student-t
# ------------------------------------------------------
plt.figure(figsize=(12,5))

# QQ-plot Normal
plt.subplot(1,2,1)
probplot(residuals, dist="norm", plot=plt)
plt.title("QQ-plot vs Normal")

# QQ-plot Student-t
plt.subplot(1,2,2)
# Use frozen t distribution for quantiles
t_dist = t(df_t, loc_t, scale_t)
theoretical_q = t_dist.ppf((np.arange(len(residuals)) + 0.5) / len(residuals))
ordered_residuals = np.sort(residuals)
plt.scatter(theoretical_q, ordered_residuals, s=12)
minv, maxv = min(theoretical_q.min(), ordered_residuals.min()), \
             max(theoretical_q.max(), ordered_residuals.max())
plt.plot([minv, maxv], [minv, maxv], 'r-', lw=2)
plt.xlabel("Theoretical Quantiles (Student-t)")
plt.ylabel("Residuals")
plt.title("QQ-plot vs Student-t")

plt.tight_layout()
plt.savefig('QQ_UA.pdf')
plt.show()
