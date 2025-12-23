import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

SQMI_TO_HA = 258.9988110336  # hectares per square mile

# ----------------------------------------------------------
# 1. Generic finder (same as minimal script)
# ----------------------------------------------------------

def find_city_entry(names, pops, areas, city_token):
    """
    Pick the UA row whose name contains city_token (case-insensitive),
    choosing the one with largest population.
    """
    token = city_token.upper()
    mask = names.astype(str).str.upper().str.contains(token, na=False)

    sub = pd.DataFrame({
        "name": names[mask],
        "pop": pops[mask],
        "area": areas[mask]
    }).dropna(subset=["pop", "area"])

    if sub.empty:
        return None

    idx = sub["pop"].idxmax()
    return sub.loc[idx, "name"], float(sub.loc[idx, "pop"]), float(sub.loc[idx, "area"])


# ----------------------------------------------------------
# 2. Unified city tracker using EXACT minimal-script logic
# ----------------------------------------------------------

def track_city_simple(city_token):
    rows = []

    # -------------------- 1950 + 1960 ----------------------
    df_50_60 = pd.read_csv("1950_1960-Table_1.csv")

    place = df_50_60.iloc[:, 3]
    pop60 = pd.to_numeric(df_50_60.iloc[:, 4].astype(str).str.replace(",", ""), errors="coerce")
    area60 = pd.to_numeric(df_50_60.iloc[:, 5].astype(str).str.replace(",", ""), errors="coerce")
    pop50 = pd.to_numeric(df_50_60.iloc[:, 8].astype(str).str.replace(",", ""), errors="coerce")
    area50 = pd.to_numeric(df_50_60.iloc[:, 9].astype(str).str.replace(",", ""), errors="coerce")

    for year, pops, areas in [(1950, pop50, area50), (1960, pop60, area60)]:
        res = find_city_entry(place, pops, areas, city_token)
        if res:
            name, pop, area = res
            rows.append({"year": year, "city_token": city_token,
                         "name": name, "population": pop, "area_sqmi": area})

    # -------------------- 1970 ----------------------
    df70 = pd.read_csv("1970-Table_1.csv")
    place70 = df70["Unnamed: 9"]
    pop70 = pd.to_numeric(df70["Unnamed: 10"].astype(str).str.replace(",", ""), errors="coerce")
    area70 = pd.to_numeric(df70["Unnamed: 11"].astype(str).str.replace(",", ""), errors="coerce")

    res = find_city_entry(place70, pop70, area70, city_token)
    if res:
        name, pop, area = res
        rows.append({"year": 1970, "city_token": city_token,
                     "name": name, "population": pop, "area_sqmi": area})

    # -------------------- 1980 ----------------------
    df80 = pd.read_csv("1980-Table_1.csv")
    place80 = df80["Unnamed: 3"]
    pop80 = pd.to_numeric(df80["Unnamed: 4"].astype(str).str.replace(",", ""), errors="coerce")
    area80 = pd.to_numeric(df80["Unnamed: 5"].astype(str).str.replace(",", ""), errors="coerce")

    res = find_city_entry(place80, pop80, area80, city_token)
    if res:
        name, pop, area = res
        rows.append({"year": 1980, "city_token": city_token,
                     "name": name, "population": pop, "area_sqmi": area})

    # -------------------- 1990 (special cleaning) ----------------------
    df90 = pd.read_csv("1990-Table_1.csv")
    place90 = df90["Unnamed: 3"]

    pop90 = pd.to_numeric(df90["Unnamed: 4"].astype(str).str.replace(",", ""), errors="coerce")
    area90 = pd.to_numeric(
        df90["Unnamed: 5"]
           .astype(str)
           .str.replace(",", "", regex=False)
           .str.replace(")", "", regex=False),
        errors="coerce"
    )

    res = find_city_entry(place90, pop90, area90, city_token)
    if res:
        name, pop, area = res
        rows.append({"year": 1990, "city_token": city_token,
                     "name": name, "population": pop, "area_sqmi": area})

    # -------------------- 2000 + 2010 ----------------------
    df00_10 = pd.read_csv("2000-2010-Table_1.csv")

    # 2010
    place10 = df00_10["Unnamed: 2"]
    pop10 = pd.to_numeric(df00_10["Unnamed: 5"].astype(str).str.replace(",", ""), errors="coerce")
    area10 = pd.to_numeric(df00_10["Unnamed: 7"].astype(str).str.replace(",", ""), errors="coerce")

    # 2000
    place00 = df00_10["Unnamed: 4"]
    pop00 = pd.to_numeric(df00_10["Unnamed: 6"].astype(str).str.replace(",", ""), errors="coerce")
    area00 = pd.to_numeric(df00_10["Unnamed: 8"].astype(str).str.replace(",", ""), errors="coerce")

    for year, places, pops, areas in [(2000, place00, pop00, area00),
                                      (2010, place10, pop10, area10)]:
        res = find_city_entry(places, pops, areas, city_token)
        if res:
            name, pop, area = res
            rows.append({"year": year, "city_token": city_token,
                         "name": name, "population": pop, "area_sqmi": area})

    # -------------------- 2020 ----------------------
    df20 = pd.read_csv("2020-Table_1.csv")
    place20 = df20["Unnamed: 1"]
    pop20 = pd.to_numeric(df20["Unnamed: 2"].astype(str).str.replace(",", ""), errors="coerce")
    area20 = pd.to_numeric(df20["Unnamed: 4"].astype(str).str.replace(",", ""), errors="coerce")

    res = find_city_entry(place20, pop20, area20, city_token)
    if res:
        name, pop, area = res
        rows.append({"year": 2020, "city_token": city_token,
                     "name": name, "population": pop, "area_sqmi": area})

    # -------------------- Final DataFrame ----------------------
    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df["area_ha"] = df["area_sqmi"] * SQMI_TO_HA
    df["m2_per_person"] = df["area_ha"] * 1e4 / df["population"]
    return df


# ----------------------------------------------------------
# 3. Plotting functions: per-city (A) and cross-city (B)
# ----------------------------------------------------------

def plot_city_evolution(df_city, city_token, outdir="city_plots"):
    """
    A: 3-panel plot for one city:
        1) Population (millions)
        2) Area (hectares)
        3) m² per person
    """
    if df_city.empty:
        print(f"[WARN] No data to plot for {city_token}")
        return

    os.makedirs(outdir, exist_ok=True)

    years = df_city["year"].values

    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # Population
    ax[0].plot(years, df_city["population"].values / 1e6, "o-")
    ax[0].set_ylabel("Population (millions)")
    ax[0].set_title(f"{city_token} – Population over time")
    #ax[0].grid(alpha=0.3)

    # Area (ha)
    ax[1].plot(years, df_city["area_ha"].values, "o-", color="green")
    ax[1].set_ylabel("Area (hectares)")
    ax[1].set_title(f"{city_token} – Urbanized land area")
    #ax[1].grid(alpha=0.3)

    # Land per person (m²)
    ax[2].plot(years, df_city["m2_per_person"].values, "o-", color="red")
    ax[2].set_ylabel("m² per person")
    ax[2].set_xlabel("Year")
    ax[2].set_title(f"{city_token} – Land area per person")
    #ax[2].grid(alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(outdir, f"{city_token.replace(' ', '_')}_evolution.png")
    plt.savefig(fname, dpi=200)
    plt.show()
    print(f"[INFO] Saved {fname}")


def plot_cross_city(panel, ordered_tokens, outdir="city_plots"):
    """
    B: Cross-city comparison plots:
        - Population vs year
        - Area vs year
        - m² per person vs year
    """
    if panel.empty:
        print("[WARN] Panel is empty, nothing to plot.")
        return

    os.makedirs(outdir, exist_ok=True)

    #cities = sorted(panel["city_token"].unique())
    cities = ordered_tokens  # already sorted by 2020 population
    cmap = plt.get_cmap("tab20")
    colors = {city: cmap(i % 20) for i, city in enumerate(cities)}

    # 1) Population
    plt.figure(figsize=(9, 6))
    for city in cities:
        sub = panel[panel["city_token"] == city].sort_values("year")
        plt.plot(sub["year"], sub["population"] / 1e6, "-o",
                 label=city, color=colors[city])
    plt.xlabel("Year")
    plt.ylabel("Population (millions)")
    plt.title("Top 25 UAs – Population over time")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    fname = os.path.join(outdir, "top25_population_over_time.png")
    plt.savefig(fname, dpi=200)
    plt.show()
    print(f"[INFO] Saved {fname}")

    # 2) Area (ha)
    plt.figure(figsize=(9, 6))
    for city in cities:
        sub = panel[panel["city_token"] == city].sort_values("year")
        plt.plot(sub["year"], sub["area_ha"], "-o",
                 label=city, color=colors[city])
    plt.xlabel("Year")
    plt.ylabel("Area (hectares)")
    plt.title("Top 25 UAs – Urbanized land area over time")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    fname = os.path.join(outdir, "top25_area_over_time.png")
    plt.savefig(fname, dpi=200)
    plt.show()
    print(f"[INFO] Saved {fname}")

    # 3) m² per person
    plt.figure(figsize=(9, 6))
    for city in cities:
        sub = panel[panel["city_token"] == city].sort_values("year")
        plt.plot(sub["year"], sub["m2_per_person"], "-o",
                 label=city, color=colors[city])
    plt.xlabel("Year")
    plt.ylabel("m² per person")
    plt.title("Top 25 UAs – Land per person over time")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    fname = os.path.join(outdir, "top25_m2_per_person_over_time.png")
    plt.savefig(fname, dpi=200)
    plt.show()
    print(f"[INFO] Saved {fname}")


# ----------------------------------------------------------
# 4. Build the Top 25 list from 2020
# ----------------------------------------------------------

def extract_city_token(name):
    """
    Extract main city name from 2020 UA label.
    """
    s = str(name)
    for sep in ["–", "-"]:
        if sep in s:
            s = s.split(sep)[0]
            break
    if "," in s:
        s = s.split(",")[0]
    return s.strip()


def main():
    df20 = pd.read_csv("2020-Table_1.csv")
    names20 = df20["Unnamed: 1"]
    pop20 = pd.to_numeric(df20["Unnamed: 2"].astype(str).str.replace(",", ""), errors="coerce")

    top20 = pd.DataFrame({"name": names20, "pop": pop20}).dropna()
    top25 = top20.sort_values("pop", ascending=False).head(25).copy()
    top25["token"] = top25["name"].apply(extract_city_token)

    print("\n[INFO] Top 25 tokens:")
    print(top25[["name", "token"]])

    full_panel = []

    # A: per-city plots
    for token in top25["token"]:
        print(f"\n=====================\nTracking: {token}\n=====================\n")
        df_city = track_city_simple(token)
        print(df_city)
        full_panel.append(df_city)
        plot_city_evolution(df_city, token)  # per-city A plots

    panel = pd.concat(full_panel, ignore_index=True)
    panel.to_csv("top25_panel_simple.csv", index=False)
    print("\n[INFO] Saved top25_panel_simple.csv")

    # B: cross-city comparison plots
    #plot_cross_city(panel)
    ordered_tokens = list(top25["token"])
    plot_cross_city(panel, ordered_tokens)


if __name__ == "__main__":
    main()
