import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from adjustText import adjust_text   # <-- NEW

# ----------------------------------------------------------
# Load panel
# ----------------------------------------------------------
panel = pd.read_csv("top200_panel_simple_2.csv")

# Keep only cities with full 8 years of data
full_cities = (
    panel.groupby("city_token")["year"].nunique()
          .pipe(lambda s: s[s == 8].index.tolist())
)

panel_full = panel[panel["city_token"].isin(full_cities)].copy()

# Remove duplicate (city_token, year) rows if any
panel_full = panel_full.drop_duplicates(subset=["city_token", "year"], keep="first")

# Pivot to wide format for m²/person
df_wide = panel_full.pivot(index="city_token",
                           columns="year",
                           values="m2_per_person")
df_wide = df_wide.sort_index(axis=1)

years = np.array(df_wide.columns)
logX = np.log(df_wide.values)

# ----------------------------------------------------------
# Compute slopes: early = 1950→1990, late = 1990→2020
# ----------------------------------------------------------
def linear_slope(x, t):
    coeffs = np.polyfit(t, x, 1)
    return coeffs[0]

mask_early = (years <= 1990)
mask_late  = (years >= 1990)

slopes_early = []
slopes_late = []

for i, city in enumerate(df_wide.index):
    xi = logX[i, :]

    s_early = linear_slope(xi[mask_early], years[mask_early])
    s_late  = linear_slope(xi[mask_late],  years[mask_late])

    slopes_early.append(s_early)
    slopes_late.append(s_late)

slopes = pd.DataFrame({
    "city_token": df_wide.index,
    "slope_early": slopes_early,
    "slope_late": slopes_late
})

# ----------------------------------------------------------
# Cluster slopes (k = 3)
# ----------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=0, n_init=20)
labels = kmeans.fit_predict(slopes[["slope_early", "slope_late"]])

slopes["cluster"] = labels

print("\nCluster counts:")
print(slopes["cluster"].value_counts().sort_index())

print("\nExample cities by cluster:")
for c in range(3):
    print(f"\nCluster {c}:")
    print(slopes[slopes["cluster"] == c].head(10))

# ----------------------------------------------------------
# Compute centered extremeness measure
# (distance from mean slope)
# ----------------------------------------------------------
mean_early = slopes["slope_early"].mean()
mean_late  = slopes["slope_late"].mean()

slopes["radius"] = np.sqrt(
    (slopes["slope_early"] - mean_early)**2 +
    (slopes["slope_late"]  - mean_late )**2
)

# ----------------------------------------------------------
# Scatter plot with adjustText labeling
# ----------------------------------------------------------
plt.figure(figsize=(8, 7))

# Color clusters
for c in range(3):
    sub = slopes[slopes["cluster"] == c]
    plt.scatter(sub["slope_early"], sub["slope_late"], label=f"Cluster {c}", s=50)

plt.axhline(0, color="gray", linewidth=1)
plt.axvline(0, color="gray", linewidth=1)
plt.xlabel("Slope (log m²/person) 1950→1990",fontsize=14)
plt.ylabel("Slope (log m²/person) 1990→2020", fontsize=14)
#plt.title("Clustering cities by early/late density slopes")
#plt.grid(True, alpha=0.3)

plt.legend()

# ----------------------------------------------------------
# Automatically label the most extreme cities
# ----------------------------------------------------------
N_LABEL = 30  # number of auto-labeled extreme cities

extreme = slopes.nlargest(N_LABEL, "radius")

texts = []

# Label extreme cities
for _, row in extreme.iterrows():
    texts.append(
        plt.text(row["slope_early"],
                 row["slope_late"],
                 row["city_token"],
                 fontsize=7)
    )

# Optionally force-label manually important metros
important = ["New York", "Chicago", "Phoenix", "Boston", "Atlanta"]

for city in important:
    if city in slopes["city_token"].values:
        r = slopes.loc[slopes["city_token"] == city].iloc[0]
        texts.append(
            plt.text(r["slope_early"],
                     r["slope_late"],
                     city,
                     fontsize=7)   # same size, no bold
        )

# Adjust label positions to avoid overlap
adjust_text(
    texts,
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    only_move={'points':'y', 'texts':'y'}  # smoother layout (optional)
)

plt.tight_layout()
plt.savefig("city_growth_clusters.pdf", dpi=200)
plt.show()

# ----------------------------------------------------------
# Save slope + cluster table
# ----------------------------------------------------------
slopes.to_csv("city_slopes_clusters.csv", index=False)
print("\nSaved city_slopes_clusters.csv")

