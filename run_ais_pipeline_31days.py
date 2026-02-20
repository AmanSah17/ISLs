import argparse
import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Process first 31 days of AIS CSV files.")
    parser.add_argument(
        "--data-dir",
        default="/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo",
        help="Directory containing daily AIS CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/processed_data",
        help="Directory to save processed parquet files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=31,
        help="Maximum number of daily files to process in sorted order.",
    )
    return parser.parse_args()


def haversine_km(lat, lon, ref_lat=32.0, ref_lon=-77.0):
    lat1 = np.radians(lat.astype(float))
    lon1 = np.radians(lon.astype(float))
    lat2 = np.radians(ref_lat)
    lon2 = np.radians(ref_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def categorize_direction(cog):
    cog = pd.to_numeric(cog, errors="coerce")
    return np.select(
        [
            (cog >= 22.5) & (cog < 67.5),
            (cog >= 67.5) & (cog < 112.5),
            (cog >= 112.5) & (cog < 157.5),
            (cog >= 157.5) & (cog < 202.5),
            (cog >= 202.5) & (cog < 247.5),
            (cog >= 247.5) & (cog < 292.5),
            (cog >= 292.5) & (cog < 337.5),
        ],
        ["NE", "E", "SE", "S", "SW", "W", "NW"],
        default="N",
    )


def process_one_file(csv_path, output_dir):
    fname = os.path.basename(csv_path)
    print(f"\n=== Processing {fname} ===")

    # Load
    df = pd.read_csv(csv_path, low_memory=False)
    drop_cols = [c for c in ["", "Unnamed: 17"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Normalize column names to pipeline style
    rename_map = {
        "mmsi": "MMSI",
        "base_date_time": "BASEDATETIME",
        "longitude": "LON",
        "latitude": "LAT",
        "sog": "SOG",
        "cog": "COG",
        "heading": "HEADING",
        "vessel_name": "VESSELNAME",
        "imo": "IMO",
        "call_sign": "CALLSIGN",
        "vessel_type": "VESSELTYPE",
        "status": "NAVSTATUS",
        "length": "LENGTH",
        "width": "WIDTH",
        "draft": "DRAFT",
        "cargo": "CARGO",
        "transceiver": "TRANSCEIVER",
    }
    df = df.rename(columns=rename_map)

    initial_rows = len(df)

    # Numeric conversions
    numeric_cols = [
        "MMSI",
        "LON",
        "LAT",
        "SOG",
        "COG",
        "HEADING",
        "VESSELTYPE",
        "NAVSTATUS",
        "LENGTH",
        "WIDTH",
        "DRAFT",
        "CARGO",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "BASEDATETIME" in df.columns:
        df["BASEDATETIME"] = pd.to_datetime(df["BASEDATETIME"], errors="coerce")

    # Cleaning / validation
    df = df.drop_duplicates(subset=["MMSI", "BASEDATETIME"], keep="first")
    df = df[df["LAT"].between(-90, 90) & df["LON"].between(-180, 180)]
    df = df[df["SOG"].between(0, 100, inclusive="both")]
    df = df[df["COG"].between(0, 360, inclusive="both")]
    df = df.dropna(subset=["MMSI", "LAT", "LON", "BASEDATETIME"])

    # Feature engineering
    ts = df["BASEDATETIME"]
    df["HOUR"] = ts.dt.hour
    df["DAY"] = ts.dt.day
    df["DAYOFWEEK"] = ts.dt.dayofweek
    df["MONTH"] = ts.dt.month
    df["WEEK"] = ts.dt.isocalendar().week.astype("Int64")
    df["ISWEEKEND"] = df["DAYOFWEEK"].isin([5, 6]).astype(int)

    df["TIMEOFDAY"] = np.select(
        [
            (df["HOUR"] >= 6) & (df["HOUR"] < 12),
            (df["HOUR"] >= 12) & (df["HOUR"] < 18),
            (df["HOUR"] >= 18) & (df["HOUR"] < 24),
        ],
        ["Morning", "Afternoon", "Evening"],
        default="Night",
    )

    df["SPEED_CATEGORY"] = pd.cut(
        df["SOG"],
        bins=[0, 5, 15, 25, 100],
        labels=["Stationary", "Slow", "Medium", "Fast"],
        include_lowest=True,
    )

    gulf = (df["LAT"] > 24) & (df["LAT"] < 31) & (df["LON"] > -87) & (df["LON"] < -80)
    northeast = (df["LAT"] > 39) & (df["LAT"] < 45) & (df["LON"] > -76) & (df["LON"] < -70)
    midatlantic = (df["LAT"] > 33) & (df["LAT"] < 36) & (df["LON"] > -79) & (df["LON"] < -75)
    pacific_nw = (df["LAT"] > 47) & (df["LAT"] < 50) & (df["LON"] > -125) & (df["LON"] < -120)
    df["MARITIMEZONE"] = np.select(
        [gulf, northeast, midatlantic, pacific_nw],
        ["Gulf of Mexico", "Northeast Coast", "Mid-Atlantic", "Pacific Northwest"],
        default="Other",
    )

    df = df.sort_values(["MMSI", "BASEDATETIME"]).reset_index(drop=True)
    df["SPEEDCHANGE"] = df.groupby("MMSI")["SOG"].diff().fillna(0.0)
    course_change = df.groupby("MMSI")["COG"].diff().abs()
    df["COURSECHANGE"] = np.where(course_change.notna(), np.minimum(course_change, 360 - course_change), 0.0)
    df["DIRECTIONQUADRANT"] = categorize_direction(df["COG"]).astype(str)
    df["DISTFROMREFERENCE"] = haversine_km(df["LAT"].values, df["LON"].values)

    # Missing values
    num_cols_all = df.select_dtypes(include=[np.number]).columns
    for col in num_cols_all:
        if df[col].isna().any():
            if "SOG" in col or "SPEED" in col:
                df[col] = df[col].fillna(df[col].median())
            elif "COG" in col or "COURSE" in col:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode_val)
            else:
                df[col] = df[col].ffill().bfill()

    # Outlier clipping
    for col in ["SOG", "COG", "SPEEDCHANGE"]:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lb = q1 - 3 * iqr
            ub = q3 + 3 * iqr
            df[col] = df[col].clip(lb, ub)

    # Normalization and encoding
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_scaling = ["MMSI", "IMO", "HOUR", "DAY", "DAYOFWEEK", "MONTH", "WEEK", "ISWEEKEND"]
    to_scale = [c for c in numeric_cols if c not in exclude_scaling]
    if to_scale:
        scaler = MinMaxScaler()
        df[to_scale] = scaler.fit_transform(df[to_scale])

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if col == "BASEDATETIME":
            continue
        df[f"{col}_ENCODED"] = pd.factorize(df[col].astype(str))[0]

    for col in ["TIMEOFDAY", "MARITIMEZONE", "DIRECTIONQUADRANT"]:
        if col in df.columns:
            onehot = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, onehot], axis=1)

    # Export
    day_match = fname.replace("ais-", "").replace(".csv", "")
    day_output = os.path.join(output_dir, "daily")
    os.makedirs(day_output, exist_ok=True)
    out_path = os.path.join(day_output, f"processed_{day_match}.parquet")
    df.to_parquet(out_path, index=False, compression="snappy")

    summary = {
        "file": fname,
        "input_rows": initial_rows,
        "output_rows": len(df),
        "output_cols": df.shape[1],
        "output_path": out_path,
    }
    print(
        f"✓ Done {fname}: {initial_rows:,} -> {len(df):,} rows, "
        f"{df.shape[1]} cols"
    )
    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pattern = os.path.join(args.data_dir, "ais-2025-01-*.csv")
    files = sorted(glob.glob(pattern))
    files = files[: args.max_files]

    if not files:
        raise FileNotFoundError(f"No files found with pattern: {pattern}")

    print(f"Found {len(files)} files to process.")
    start = datetime.now()

    summaries = []
    for path in files:
        summaries.append(process_one_file(path, args.output_dir))

    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join(args.output_dir, "processing_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Processed files: {len(files)}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Total input rows: {summary_df['input_rows'].sum():,}")
    print(f"Total output rows: {summary_df['output_rows'].sum():,}")
    print(f"Elapsed: {datetime.now() - start}")


if __name__ == "__main__":
    main()
