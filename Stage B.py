from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "Data_Centre_#13_2025.csv"

# Reference capacity of the data centre.
# This is a modelling assumption because UKPN anonymised MIC.
P_REF_MW = 10.0

# Assumed power factor to convert apparent power (kVA) to
# active power (kW).
POWER_FACTOR = 0.95

# Expected year
YEAR = 2025

# Expected number of half-hourly and hourly points
EXPECTED_HALF_HOURLY = 365 * 48      # 17,520
EXPECTED_HOURLY = 365 * 24            # 8,760


# ============================================================
# 2. OUTPUT FILES
# ============================================================

OUTPUT_HOURLY = BASE_DIR / "Data_Centre_#13_2025_hourly.csv"
OUTPUT_QUALITY = BASE_DIR / "Data_Centre_#13_2025_quality_report.txt"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 60)
print("DATA CENTRE LOAD PROFILE PROCESSING")
print("=" * 60)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["utc_timestamp"]
)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# 4. BASIC CHECKS
# ============================================================

required_columns = [
    "utc_timestamp",
    "hh_utilisation_ratio"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Keep only the variables needed for the load profile
data = df[
    [
        "utc_timestamp",
        "hh_utilisation_ratio"
    ]
].copy()


# Sort chronologically
data = data.sort_values("utc_timestamp")


# ============================================================
# 5. QUALITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("QUALITY CHECK")
print("=" * 60)


# ------------------------------------------------------------
# 5.1 Check year
# ------------------------------------------------------------

data["year"] = data["utc_timestamp"].dt.year

years_found = sorted(data["year"].dropna().unique())

print(f"\nYears found: {years_found}")

if YEAR not in years_found:
    raise ValueError(
        f"No data found for year {YEAR}."
    )

data = data[data["year"] == YEAR].copy()

data = data.drop(columns=["year"])


# ------------------------------------------------------------
# 5.2 Check missing timestamps
# ------------------------------------------------------------

expected_30min = pd.date_range(
    start=f"{YEAR}-01-01 00:00:00",
    end=f"{YEAR}-12-31 23:30:00",
    freq="30min"
)

actual_timestamps = data["utc_timestamp"]

missing_timestamps = expected_30min.difference(
    actual_timestamps
)

extra_timestamps = actual_timestamps[
    ~actual_timestamps.isin(expected_30min)
].unique()


# ------------------------------------------------------------
# 5.3 Check duplicate timestamps
# ------------------------------------------------------------

duplicate_mask = data["utc_timestamp"].duplicated(
    keep=False
)

duplicate_rows = data[duplicate_mask].copy()

n_duplicate_rows = len(duplicate_rows)

n_duplicate_timestamps = (
    duplicate_rows["utc_timestamp"]
    .nunique()
    if n_duplicate_rows > 0
    else 0
)


# ------------------------------------------------------------
# 5.4 Check number of rows
# ------------------------------------------------------------

n_rows = len(data)

n_unique_timestamps = data["utc_timestamp"].nunique()


print(f"\nRows:                 {n_rows:,}")
print(f"Unique timestamps:    {n_unique_timestamps:,}")
print(f"Expected timestamps:  {EXPECTED_HALF_HOURLY:,}")

print(f"\nMissing timestamps:   {len(missing_timestamps):,}")
print(f"Duplicate rows:       {n_duplicate_rows:,}")
print(f"Duplicate timestamps: {n_duplicate_timestamps:,}")
print(f"Extra timestamps:     {len(extra_timestamps):,}")


# ------------------------------------------------------------
# 5.5 Check utilisation values
# ------------------------------------------------------------

util = data["hh_utilisation_ratio"]

n_missing_util = util.isna().sum()
n_negative_util = (util < 0).sum()
n_over_1_util = (util > 1).sum()

print(f"\nMissing utilisation:   {n_missing_util:,}")
print(f"Negative utilisation:  {n_negative_util:,}")
print(f"Utilisation > 1:      {n_over_1_util:,}")

print("\nUtilisation statistics:")
print(util.describe())


# ------------------------------------------------------------
# 5.6 Check 30-minute frequency
# ------------------------------------------------------------

timestamp_diff = (
    data["utc_timestamp"]
    .sort_values()
    .diff()
)

unexpected_intervals = timestamp_diff[
    timestamp_diff.notna() &
    (timestamp_diff != pd.Timedelta(minutes=30))
]

print(
    f"\nUnexpected time intervals: "
    f"{len(unexpected_intervals):,}"
)


# ============================================================
# 6. STOP IF DATA QUALITY IS NOT SUITABLE
# ============================================================

if (
    len(missing_timestamps) > 0
    or n_duplicate_timestamps > 0
    or n_missing_util > 0
    or n_negative_util > 0
    or len(unexpected_intervals) > 0
):

    print("\nWARNING:")
    print(
        "The half-hourly dataset is not perfectly continuous."
    )
    print(
        "The script will NOT interpolate or fill missing data."
    )

    print(
        "\nYou should inspect the quality issues before "
        "using this profile for BHESS."
    )

    # Save quality report
    with open(OUTPUT_QUALITY, "w", encoding="utf-8") as f:

        f.write("DATA CENTRE QUALITY REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Year: {YEAR}\n")
        f.write(f"Rows: {n_rows}\n")
        f.write(
            f"Unique timestamps: "
            f"{n_unique_timestamps}\n"
        )
        f.write(
            f"Expected timestamps: "
            f"{EXPECTED_HALF_HOURLY}\n"
        )

        f.write(
            f"\nMissing timestamps: "
            f"{len(missing_timestamps)}\n"
        )

        f.write(
            f"Duplicate timestamps: "
            f"{n_duplicate_timestamps}\n"
        )

        f.write(
            f"Missing utilisation: "
            f"{n_missing_util}\n"
        )

        f.write(
            f"Negative utilisation: "
            f"{n_negative_util}\n"
        )

        f.write(
            f"Utilisation > 1: "
            f"{n_over_1_util}\n"
        )

        f.write(
            f"Unexpected intervals: "
            f"{len(unexpected_intervals)}\n"
        )

        if len(missing_timestamps) > 0:

            f.write("\nMISSING TIMESTAMPS\n")
            f.write("-" * 60 + "\n")

            for ts in missing_timestamps:
                f.write(str(ts) + "\n")

    print(
        f"\nQuality report saved to:\n"
        f"{OUTPUT_QUALITY}"
    )

    raise SystemExit(
        "\nProcessing stopped because the input data "
        "requires inspection."
    )


# ============================================================
# 7. HALF-HOURLY → HOURLY
# ============================================================

print("\n" + "=" * 60)
print("30-MINUTE → HOURLY")
print("=" * 60)


data = data.set_index("utc_timestamp")


hourly = (
    data["hh_utilisation_ratio"]
    .resample("1h")
    .mean()
    .to_frame(name="hourly_utilisation_ratio")
)


print(
    f"\nHourly points generated: "
    f"{len(hourly):,}"
)

if len(hourly) != EXPECTED_HOURLY:
    raise ValueError(
        f"Expected {EXPECTED_HOURLY} hourly points, "
        f"but obtained {len(hourly)}."
    )


# ============================================================
# 8. DEFINE REFERENCE CAPACITY
# ============================================================

print("\n" + "=" * 60)
print("LOAD SCALING")
print("=" * 60)

P_REF_KW = P_REF_MW * 1000

print(f"\nReference capacity: {P_REF_MW:.2f} MW")
print(f"Reference capacity: {P_REF_KW:,.0f} kW")

print(f"Power factor:       {POWER_FACTOR:.3f}")


# ============================================================
# 9. CONVERT UTILISATION → APPARENT POWER
# ============================================================

# UKPN utilisation is based on apparent power:
#
# S_t = utilisation_t × Maximum Import Capacity
#
# Since the actual MIC is anonymised, P_REF_MW is used
# as a modelling reference capacity.

hourly["apparent_power_kVA"] = (
    hourly["hourly_utilisation_ratio"]
    * P_REF_KW
)


# ============================================================
# 10. CONVERT kVA → kW
# ============================================================

# P = S × power factor

hourly["load_kW"] = (
    hourly["apparent_power_kVA"]
    * POWER_FACTOR
)


# ============================================================
# 11. CONVERT kW → kWh
# ============================================================

# Timestep = 1 hour
#
# Energy = Power × time
#
# E[kWh] = P[kW] × 1[h]

hourly["load_kWh"] = (
    hourly["load_kW"] * 1.0
)


# ============================================================
# 12. ADD USEFUL INFORMATION FOR BHESS
# ============================================================

hourly["hour"] = np.arange(
    1,
    len(hourly) + 1
)


# Reorder columns
hourly = hourly[
    [
        "hour",
        "hourly_utilisation_ratio",
        "apparent_power_kVA",
        "load_kW",
        "load_kWh"
    ]
]


# ============================================================
# 13. FINAL 8760-HOUR QUALITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL 8760-HOUR PROFILE CHECK")
print("=" * 60)

print(f"\nNumber of hours: {len(hourly):,}")

print(
    f"Expected hours:  {EXPECTED_HOURLY:,}"
)

print(
    f"Minimum load:    {hourly['load_kW'].min():,.2f} kW"
)

print(
    f"Maximum load:    {hourly['load_kW'].max():,.2f} kW"
)

print(
    f"Mean load:       {hourly['load_kW'].mean():,.2f} kW"
)

print(
    f"Annual energy:   {hourly['load_kWh'].sum():,.2f} kWh"
)

print(
    f"Annual energy:   "
    f"{hourly['load_kWh'].sum() / 1000:,.2f} MWh"
)

print(
    f"Load factor:     "
    f"{hourly['load_kW'].mean() / hourly['load_kW'].max():.4f}"
)


# ============================================================
# 14. EXPORT
# ============================================================

hourly.to_csv(
    OUTPUT_HOURLY,
    index=False
)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(
    f"\nHourly BHESS load profile saved to:\n"
    f"{OUTPUT_HOURLY}"
)

print("\nFirst 10 rows:")
print(hourly.head(10))

print("\nLast 10 rows:")
print(hourly.tail(10))