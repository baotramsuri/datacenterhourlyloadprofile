import pandas as pd
from pathlib import Path

# ============================================================
# 1. File
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
csv_file = BASE_DIR / "ukpn-data-centre-demand-profiles.csv"

# ============================================================
# 2. Read only required columns
# ============================================================

usecols = [
    "anonymised_data_centre_name",
    "utc_timestamp"
]

records = []

for chunk in pd.read_csv(
    csv_file,
    usecols=usecols,
    parse_dates=["utc_timestamp"],
    chunksize=100_000
):
    chunk["year"] = chunk["utc_timestamp"].dt.year

    # Count ROWS
    row_count = (
        chunk
        .groupby(["anonymised_data_centre_name", "year"])
        .size()
        .reset_index(name="n_rows")
    )

    # Count UNIQUE timestamps within this chunk
    unique_count = (
        chunk
        .drop_duplicates(
            ["anonymised_data_centre_name", "utc_timestamp"]
        )
        .groupby(["anonymised_data_centre_name", "year"])
        .size()
        .reset_index(name="n_unique_timestamps")
    )

    x = row_count.merge(
        unique_count,
        on=["anonymised_data_centre_name", "year"],
        how="outer"
    )

    records.append(x)

# ============================================================
# 3. Combine chunks
# ============================================================

result = pd.concat(records, ignore_index=True)

# Total rows across chunks
rows = (
    result
    .groupby(
        ["anonymised_data_centre_name", "year"],
        as_index=False
    )["n_rows"]
    .sum()
)

# IMPORTANT:
# Unique timestamps cannot simply be summed across chunks
# because the same timestamp could occur in different chunks.
#
# Therefore, do a second pass for exact unique timestamp count.
# ============================================================

unique_timestamps = {}

for chunk in pd.read_csv(
    csv_file,
    usecols=usecols,
    parse_dates=["utc_timestamp"],
    chunksize=100_000
):
    chunk["year"] = chunk["utc_timestamp"].dt.year

    for (site, year), group in chunk.groupby(
        ["anonymised_data_centre_name", "year"]
    ):
        key = (site, year)

        if key not in unique_timestamps:
            unique_timestamps[key] = set()

        unique_timestamps[key].update(
            group["utc_timestamp"].dropna()
        )

# ============================================================
# 4. Create final table
# ============================================================

final = rows.copy()

final["n_unique_timestamps"] = [
    len(unique_timestamps.get(
        (site, year), set()
    ))
    for site, year in zip(
        final["anonymised_data_centre_name"],
        final["year"]
    )
]

def expected_points(year):
    return (
        366 * 48
        if pd.Timestamp(year=year, month=1, day=1).is_leap_year
        else 365 * 48
    )

final["expected"] = final["year"].apply(expected_points)

final["complete"] = (
    final["n_unique_timestamps"]
    == final["expected"]
)

# ============================================================
# 5. Show 2025
# ============================================================

result_2025 = (
    final[final["year"] == 2025]
    .sort_values("n_unique_timestamps", ascending=False)
)

print("\n2025:")
print(result_2025.to_string(index=False))

# ============================================================
# 6. Show only genuinely complete sites
# ============================================================

complete_2025 = result_2025[
    result_2025["complete"]
]

print("\nCOMPLETE 2025 DATA CENTRES:")
print(
    complete_2025.to_string(index=False)
)
# ============================================================
# 7. Select data centre and year
# ============================================================

selected_site = "Data Centre #13"
selected_year = 2025

# ============================================================
# 8. Extract selected data centre and year
# ============================================================

output_file = (
    BASE_DIR /
    f"{selected_site.replace(' ', '_')}_{selected_year}.csv"
)

first_chunk = True
total_rows = 0

for chunk in pd.read_csv(
    csv_file,
    parse_dates=["utc_timestamp"],
    chunksize=100_000
):

    site_year = chunk[
        (chunk["anonymised_data_centre_name"] == selected_site) &
        (chunk["utc_timestamp"].dt.year == selected_year)
    ]

    if not site_year.empty:

        site_year = site_year.sort_values("utc_timestamp")

        site_year.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False
        total_rows += len(site_year)

# ============================================================
# 9. Result
# ============================================================

print(f"\nExported: {output_file}")
print(f"Data Centre: {selected_site}")
print(f"Year: {selected_year}")
print(f"Rows: {total_rows}")