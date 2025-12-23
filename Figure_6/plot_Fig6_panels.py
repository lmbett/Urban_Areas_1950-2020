import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -------------------------------------------------
# File path
# -------------------------------------------------
DATA_FILE = "A0_VMTpc_congestionGDP_decadal.csv"

# -------------------------------------------------
# Load data
# -------------------------------------------------
df = pd.read_csv(DATA_FILE)

print("Columns found in file:")
print(df.columns.tolist())

# -------------------------------------------------
# Convert numeric columns safely
# -------------------------------------------------
num_cols = [
    "year",
    "A0_all",
    "A0_fixed_all",
    "A0_all_50k",
    "vmt_per_capita",
    "GDP ($billion)",
    "Congestion Costs ($million)",
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values("year")

# -------------------------------------------------
# Construct inverse congestion metric
# GDP in billions, congestion in millions
# -------------------------------------------------
df["GDP_per_congestion"] = (
    (df["GDP ($billion)"] * 1e3) / df["Congestion Costs ($million)"]
)

# -------------------------------------------------
# Log variables (for elasticities)
# -------------------------------------------------
df["log_VMT"] = np.log(df["vmt_per_capita"])

df["log_A0_all"] = np.log(df["A0_all"])
df["log_A0_fixed"] = np.log(df["A0_fixed_all"])
df["log_A0_50k"] = np.log(df["A0_all_50k"])

df["log_GDP_per_cong"] = np.log(df["GDP_per_congestion"])

# -------------------------------------------------
# Helper: log–log regression
# -------------------------------------------------
def log_log_reg(y, x, label):
    mask = y.notna() & x.notna()
    yv = y[mask]
    xv = x[mask]

    if len(yv) < 3:
        print(f"\n[SKIP] {label}: insufficient data (n={len(yv)})")
        return None

    X = sm.add_constant(xv)
    model = sm.OLS(yv, X).fit()

    beta = model.params.iloc[1]
    ci = model.conf_int().iloc[1]

    print("\n==============================")
    print(label)
    print("==============================")
    print(f"n = {len(yv)}")
    print(f"Elasticity = {beta:.3f}")
    print(f"95% CI     = [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"R²         = {model.rsquared:.3f}")
    print(f"AIC        = {model.aic:.2f}")

    return model

# =================================================
# ELASTICITY ANALYSIS
# =================================================

print("\n============================================")
print("LOG–LOG ELASTICITIES: A0 vs VMT per capita")
print("============================================")

log_log_reg(
    df["log_A0_all"],
    df["log_VMT"],
    "log(A0_all) ~ log(VMT per capita)"
)

log_log_reg(
    df["log_A0_fixed"],
    df["log_VMT"],
    "log(A0_fixed) ~ log(VMT per capita)"
)

log_log_reg(
    df["log_A0_50k"],
    df["log_VMT"],
    "log(A0_all_50k) ~ log(VMT per capita)"
)

print("\n============================================")
print("LOG–LOG ELASTICITIES: A0 vs GDP / congestion")
print("(post-1980 only, by construction)")
print("============================================")

log_log_reg(
    df["log_A0_all"],
    df["log_GDP_per_cong"],
    "log(A0_all) ~ log(GDP / congestion cost)"
)

log_log_reg(
    df["log_A0_fixed"],
    df["log_GDP_per_cong"],
    "log(A0_fixed) ~ log(GDP / congestion cost)"
)

log_log_reg(
    df["log_A0_50k"],
    df["log_GDP_per_cong"],
    "log(A0_all_50k) ~ log(GDP / congestion cost)"
)

# =================================================
# PLOTS 
# =================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
})

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(7.2, 8.0), sharex=True
)

# ---------------- TOP PANEL ----------------
ax_top.plot(df["year"], df["A0_all"], "-o", lw=2.5, ms=5, label=r"$A_0$ (free $b$)")
ax_top.plot(df["year"], df["A0_fixed_all"], "-s", lw=2.5, ms=5, label=r"$A_0$ ($b=5/6$ fixed)")

mask_50k = df["A0_all_50k"].notna()
ax_top.plot(
    df.loc[mask_50k, "year"],
    df.loc[mask_50k, "A0_all_50k"],
    "D", ms=7, color="black", alpha=0.3,label=r"$A_0$ (pop $>$ 50k)"
)

ax_top.set_ylabel(r"$A_0$ (hectares)")
ax_top.legend(loc="upper left", frameon=False)

ax_top_r = ax_top.twinx()
ax_top_r.plot(df["year"], df["vmt_per_capita"], "--", lw=2.3, color="tab:red", label="VMT per capita")
ax_top_r.set_ylabel("Vehicle miles traveled per capita")
ax_top_r.legend(loc="lower right", frameon=False)

#ax_top.set_title("Baseline urban area and mobility over time")

# ---------------- BOTTOM PANEL ----------------

ax_bot.plot(df["year"], df["A0_all"], "-o", lw=2.5, ms=5)
ax_bot.plot(df["year"], df["A0_fixed_all"], "-s", lw=2.5, ms=5 )


ax_bot.plot(
    df.loc[mask_50k, "year"],
    df.loc[mask_50k, "A0_all_50k"],
    "D", ms=7, color="black",alpha=0.3)


ax_bot.set_ylabel(r"$A_0$ (hectares)")
ax_bot.legend(loc="upper left", frameon=False)

ax_bot_r = ax_bot.twinx()
mask_cong = df["GDP_per_congestion"].notna()
ax_bot_r.plot(
    df.loc[mask_cong, "year"],
    df.loc[mask_cong, "GDP_per_congestion"],
    "--o", lw=2.3, ms=5, color="tab:green",label="GDP / congestion cost"
)
ax_bot_r.set_ylabel("GDP / congestion cost")
ax_bot_r.legend(loc="upper left", frameon=False)

ax_bot.axvspan(1980, 2000, color="grey", alpha=0.15, zorder=0)
ax_bot.set_xlabel("Year")

plt.tight_layout()
plt.savefig("A0_mobility_congestion_over_time.pdf")
plt.show()

# =================================================
# log(A0) vs log(VMT) and log(GDP / congestion)
# =================================================

#fig2, (ax1, ax2) = plt.subplots(
#    1, 2, figsize=(10.5, 4.2), sharey=True
#)


fig2, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7.2, 8), sharey=True
)

# -------------------------------------------------
# Style definitions 
# -------------------------------------------------
STYLE = {
    "all": dict(color="tab:blue", marker="o", alpha=0.95, s=40),
    "50k": dict(color="black", marker="D", alpha=0.3, s=80),
}

LINEWIDTH_FIT = 2.6

def add_loglog_fit(ax, x, y, color):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    xs = np.linspace(x.min(), x.max(), 100)
    ys = model.params.iloc[0] + model.params.iloc[1] * xs
    ax.plot(xs, ys, color=color, lw=LINEWIDTH_FIT)
    return model

# =================================================
# PANEL 1: log(A0) vs log(VMT per capita)
# =================================================
mask_vmt_all = df["log_A0_all"].notna() & df["log_VMT"].notna()
mask_vmt_50k = df["log_A0_50k"].notna() & df["log_VMT"].notna()

# Points
ax1.scatter(
    df.loc[mask_vmt_all, "log_VMT"],
    df.loc[mask_vmt_all, "log_A0_all"],
    label=r"$A_0$ (free $b$)",
    **STYLE["all"]
)

ax1.scatter(
    df.loc[mask_vmt_50k, "log_VMT"],
    df.loc[mask_vmt_50k, "log_A0_50k"],
    label=r"$A_0$ (pop $>$ 50k)",
    **STYLE["50k"]
)

# Fits
add_loglog_fit(
    ax1,
    df.loc[mask_vmt_all, "log_VMT"],
    df.loc[mask_vmt_all, "log_A0_all"],
    STYLE["all"]["color"]
)

if mask_vmt_50k.sum() >= 3:
    add_loglog_fit(
        ax1,
        df.loc[mask_vmt_50k, "log_VMT"],
        df.loc[mask_vmt_50k, "log_A0_50k"],
        STYLE["50k"]["color"]
    )

ax1.set_xlabel(r"$\log(\mathrm{VMT\ per\ capita})$")
ax1.set_ylabel(r"$\log(A_0\ \mathrm{[ha]})$")
#ax1.set_title("Baseline area vs mobility")

ax1.legend(frameon=False)

# =================================================
# PANEL 2: log(A0) vs log(GDP / congestion)
# =================================================
mask_cong_all = df["log_A0_all"].notna() & df["log_GDP_per_cong"].notna()
mask_cong_50k = df["log_A0_50k"].notna() & df["log_GDP_per_cong"].notna()

# Points (NO legend here)
ax2.scatter(
    df.loc[mask_cong_all, "log_GDP_per_cong"],
    df.loc[mask_cong_all, "log_A0_all"],
    **STYLE["all"]
)

ax2.scatter(
    df.loc[mask_cong_50k, "log_GDP_per_cong"],
    df.loc[mask_cong_50k, "log_A0_50k"],
    **STYLE["50k"]
)

# Fits
add_loglog_fit(
    ax2,
    df.loc[mask_cong_all, "log_GDP_per_cong"],
    df.loc[mask_cong_all, "log_A0_all"],
    STYLE["all"]["color"]
)

if mask_cong_50k.sum() >= 3:
    add_loglog_fit(
        ax2,
        df.loc[mask_cong_50k, "log_GDP_per_cong"],
        df.loc[mask_cong_50k, "log_A0_50k"],
        STYLE["50k"]["color"]
    )

ax2.set_xlabel(r"$\log(\mathrm{GDP / congestion\ cost})$")
#ax2.set_title("Baseline area vs effective mobility cost")

# -------------------------------------------------
# Final formatting
# -------------------------------------------------
plt.tight_layout()
plt.savefig("A0_loglog_elasticities_VMT_congestion_clean.pdf")
plt.show()
