from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib
import numpy as np
from pyproj import Transformer
from sklearn.neighbors import BallTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import re
import gc

def read_all_data_archive() -> dict[str, pd.DataFrame]:

    project_root = Path(__file__).resolve().parents[1]

    directories = {
        "MODIS": "MODIS_GLOABL_2025-2026",
        "SV": "SUOMI_VIIRS_C2_GLO_2025-2026",
        "J1V": "J1V-C2_GLO_2025-2026",
        "J2V": "J2V-C2_GLO_2025-2026",
    }

    dfs = {}

    for sensor, directory in directories.items():

        folder = project_root / directory

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {folder}"
            )

        # Select archive CSV files only
        archive_files = sorted(
            file for file in folder.glob("*.csv")
            if "archive" in file.stem.lower().split("_")
        )

        if not archive_files:
            continue

        # Read and concatenate all archive files
        frames = []

        for file in archive_files:

            df = pd.read_csv(file, low_memory=False)

            # Preserve original file provenance
            df["source_file"] = file.name

            frames.append(df)

        dfs[sensor] = pd.concat(
            frames,
            ignore_index=True
        )

    return dfs


def read_data_modis():
    dfs = []

    project_root = Path(__file__).resolve().parents[1]
    
    modis_files = project_root / "MODIS_GLOABL_2025-2026"

    if not modis_files.is_dir():
        raise FileNotFoundError(f"Directory not found: {modis_files}")

    for file in modis_files.glob("*.csv"):
        dfs.append(pd.read_csv(file))

    return dfs


def read_data_viirs_c2():
    dfs = []
    
    project_root = Path(__file__).resolve().parents[1]
    
    modis_files = project_root / "SUOMI_VIIRS_C2_GLO_2025-2026"

    if not modis_files.is_dir():
        raise FileNotFoundError(f"Directory not found: {modis_files}")

    for file in modis_files.glob("*.csv"):
        dfs.append(pd.read_csv(file))

    return dfs


def read_data_viirs_j1():
    dfs = []
    
    project_root = Path(__file__).resolve().parents[1]
    
    modis_files = project_root / "J1V-C2_GLO_2025-2026"

    if not modis_files.is_dir():
        raise FileNotFoundError(f"Directory not found: {modis_files}")

    for file in modis_files.glob("*.csv"):
        dfs.append(pd.read_csv(file))

    return dfs

def read_data_viirs_j2():
    dfs = []
    
    project_root = Path(__file__).resolve().parents[1]
    
    modis_files = project_root / "J2V-C2_GLO_2025-2026"

    if not modis_files.is_dir():
        raise FileNotFoundError(f"Directory not found: {modis_files}")

    for file in modis_files.glob("*.csv"):
        dfs.append(pd.read_csv(file))

    return dfs


def compare_temperature_distributions(datasets):

    frames = []

    for sensor, df in datasets.items():

        temp = df.copy()

        if sensor == "MODIS":
            temp["platform"] = temp["satellite"]

        temp["source"] = sensor

        frames.append(temp)

    combined = pd.concat(frames, ignore_index=True)

    columns = ["brightness", "bright_t31"]

    summary = (
        combined.groupby(
            ["source", "daynight"]
        )[columns]
        .agg(["count", "mean", "median", "std"])
        .round(3)
    )

    return summary



def compare_platforms_daynight(datasets):

    frames = []

    for source, df in datasets.items():

        temp = df[
            [
                "satellite",
                "daynight",
                "brightness",
                "bright_t31"
            ]
        ].copy()

        temp["source"] = source

        if source == "MODIS":
            temp["platform"] = (
                temp["satellite"].map({
                    "Aqua": "Aqua",
                    "Terra": "Terra"
                })
            )
        else:
            temp["platform"] = source

        frames.append(temp)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    summary = (
        combined.groupby(
            ["platform", "daynight"],
            observed=True
        )
        .agg(
            count=("brightness", "size"),
            mid_mean=("brightness", "mean"),
            mid_median=("brightness", "median"),
            mid_std=("brightness", "std"),
            thermal_mean=("bright_t31", "mean"),
            thermal_median=("bright_t31", "median"),
            thermal_std=("bright_t31", "std")
        )
        .round(3)
    )

    return summary



def audit_platform_coverage(datasets):

    results = []

    for source, df in datasets.items():

        temp = df[
            [
                "satellite",
                "acq_date",
                "daynight",
                "source_file"
            ]
        ].copy()

        temp["acq_date"] = pd.to_datetime(
            temp["acq_date"],
            errors="coerce"
        )

        temp["daynight"] = (
            temp["daynight"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        summary = (
            temp.groupby(
                ["satellite", "daynight"],
                dropna=False
            )
            .agg(
                detections=("acq_date", "size"),
                first_date=("acq_date", "min"),
                last_date=("acq_date", "max"),
                unique_dates=("acq_date", "nunique"),
                source_files=("source_file", "nunique")
            )
            .reset_index()
        )

        summary.insert(0, "source", source)

        results.append(summary)

    return pd.concat(results, ignore_index=True)




def categorical_mapping(
    df: pd.DataFrame,
    encoder_path: str,
    columns: list[str] | None = None,
    reference_df: pd.DataFrame | None = None
) -> pd.DataFrame:

    df = df.copy()

    encoder_path = Path(encoder_path)
    encoder_path.mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = df.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

    reference = reference_df if reference_df is not None else df

    for col in columns:

        if col not in df.columns:
            continue

        ref = reference[col] if col in reference.columns else df[col]

        # Find the mode from the reference dataset
        mode = ref.mode(dropna=True)

        if not mode.empty:
            mode_value = str(mode.iloc[0])
        else:
            mode_value = "__MISSING__"

        # Fit encoder on all unique categories in the reference data
        categories = [
            str(value) for value in ref.dropna().unique()
        ]

        if mode_value not in categories:
            categories.append(mode_value)

        if not categories:
            categories = ["__MISSING__"]

        encoder = LabelEncoder()
        encoder.fit(categories)

        # Replace missing values with the reference mode
        values = (
            df[col]
            .astype("string")
            .fillna(mode_value)
            .astype(str)
        )

        # Transform using the fitted encoder
        df[col] = encoder.transform(values)

        # Save encoder and imputation value
        joblib.dump(
            encoder,
            encoder_path / f"{col}_encoder.pkl"
        )

        joblib.dump(
            mode_value,
            encoder_path / f"{col}_mode.pkl"
        )

    return df



def build_correlation_features(
    df: pd.DataFrame,
    include_coordinates: bool = False
) -> pd.DataFrame:

    features = pd.DataFrame(index=df.index)

    def get_numeric(columns):

        for col in columns:
            if col in df.columns:
                values = pd.to_numeric(
                    df[col], errors="coerce"
                )

                if values.notna().any():
                    return values

        return pd.Series(np.nan, index=df.index)

    # 1. Brightness measurements
    features["bt_mid_ir_k"] = get_numeric(
        ["bt_mid_ir_k", "brightness"]
    )

    features["bt_thermal_k"] = get_numeric(
        ["bt_thermal_k", "bright_t31"]
    )

    features["bt_contrast_k"] = (
        features["bt_mid_ir_k"] -
        features["bt_thermal_k"]
    )

    # 2. Spatial measurements
    features["scan_km"] = get_numeric(
        ["scan_km", "scan"]
    )

    features["track_km"] = get_numeric(
        ["track_km", "track"]
    )

    features["pixel_area_km2"] = (
        features["scan_km"] *
        features["track_km"]
    )

    # 3. FRP
    features["frp_mw"] = get_numeric(
        ["frp_mw", "frp"]
    )

    features["log_frp"] = np.log1p(
        features["frp_mw"].where(
            features["frp_mw"] >= 0
        )
    )

    # 4. Confidence: explicit ordinal mapping
    confidence_rank = {
        "low": 0,
        "l": 0,
        "nominal": 1,
        "n": 1,
        "high": 2,
        "h": 2
    }

    if "confidence_class" in df.columns:

        features["confidence_rank"] = (
            df["confidence_class"]
            .astype("string")
            .str.lower()
            .map(confidence_rank)
        )

    elif "confidence" in df.columns:

        raw = df["confidence"]

        if pd.api.types.is_numeric_dtype(raw):

            values = pd.to_numeric(
                raw, errors="coerce"
            )

            features["confidence_rank"] = np.select(
                [
                    values.between(0, 30, inclusive="left"),
                    values.between(30, 80, inclusive="left"),
                    values.between(80, 100)
                ],
                [0, 1, 2],
                default=np.nan
            )

        else:

            features["confidence_rank"] = (
                raw.astype("string")
                .str.strip()
                .str.lower()
                .map(confidence_rank)
            )

    # 5. Day / night
    if "daynight" in df.columns:

        features["daynight_binary"] = (
            df["daynight"]
            .astype("string")
            .str.strip()
            .str.upper()
            .map({
                "D": 0,
                "DAY": 0,
                "N": 1,
                "NIGHT": 1
            })
        )

    # 6. Acquisition time and seasonality
    if "acq_datetime_utc" in df.columns:

        dt = pd.to_datetime(
            df["acq_datetime_utc"],
            errors="coerce",
            utc=True
        )

    else:

        dt = pd.Series(
            pd.NaT, index=df.index,
            dtype="datetime64[ns, UTC]"
        )

    if dt.notna().any():

        month_angle = (
            2 * np.pi * (dt.dt.month - 1) / 12
        )

        features["month_sin"] = np.sin(month_angle)
        features["month_cos"] = np.cos(month_angle)

        # Approximate local solar time
        utc_hour = (
            dt.dt.hour
            + dt.dt.minute / 60
            + dt.dt.second / 3600
        )

        longitude = get_numeric(["longitude"])

        local_hour = (
            utc_hour + longitude / 15
        ) % 24

        local_angle = 2 * np.pi * local_hour / 24

        features["local_solar_time_sin"] = np.sin(
            local_angle
        )

        features["local_solar_time_cos"] = np.cos(
            local_angle
        )

    # 7. Optional geographic coordinates
    if include_coordinates:

        features["latitude"] = get_numeric(
            ["latitude"]
        )

        features["longitude"] = get_numeric(
            ["longitude"]
        )

    return features.replace(
        [np.inf, -np.inf], np.nan
    )


def analyze_sensor_platform_correlations(
    df: pd.DataFrame,
    encoder_root: str = "encoder/categorical",
    correlation_root: str = "correlations",
    sample_size: int = 250_000,
    random_state: int = 42,
    include_coordinates: bool = False
):

    required = {"sensor", "platform"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    def safe_name(value):
        return re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            str(value).strip()
        )

    results = {}

    for (sensor, platform), group in df.groupby(
        ["sensor", "platform"],
        observed=True,
        sort=True
    ):

        if pd.isna(sensor) or pd.isna(platform):
            continue

        sensor_name = safe_name(sensor)
        platform_name = safe_name(platform)

        print(
            f"\nAnalyzing {sensor_name} / {platform_name}"
        )

        # ----------------------------------------
        # 1. Prepare categorical encoders
        # ----------------------------------------

        categorical_columns = [
            col for col in [
                "confidence_class",
                "daynight",
                "fire_type"
            ]
            if col in group.columns
        ]

        # Support raw VIIRS confidence if no harmonized class exists
        if (
            "confidence_class" not in group.columns
            and "confidence" in group.columns
            and not pd.api.types.is_numeric_dtype(
                group["confidence"]
            )
        ):
            categorical_columns.append("confidence")

        encoder_path = (
            Path(encoder_root)
            / sensor_name
            / platform_name
        )

        if categorical_columns:

            # Small sample for transformation;
            # reference_df provides all platform categories.
            encoder_n = min(sample_size, len(group))

            encoder_sample = group[
                categorical_columns
            ].sample(
                n=encoder_n,
                random_state=random_state
            )

            categorical_mapping(
                encoder_sample,
                encoder_path,
                columns=categorical_columns,
                reference_df=group[categorical_columns]
            )

        # ----------------------------------------
        # 2. Define analysis subsets
        # ----------------------------------------

        subsets = {"ALL": group}

        if "daynight" in group.columns:

            daynight = (
                group["daynight"]
                .astype("string")
                .str.strip()
                .str.upper()
            )

            subsets["DAY"] = group.loc[
                daynight.isin(["D", "DAY"])
            ]

            subsets["NIGHT"] = group.loc[
                daynight.isin(["N", "NIGHT"])
            ]

        # ----------------------------------------
        # 3. Calculate correlation matrices
        # ----------------------------------------

        results[(sensor_name, platform_name)] = {}

        for period, subset in subsets.items():

            if subset.empty:
                continue

            n = min(sample_size, len(subset))

            sample = subset.sample(
                n=n,
                random_state=random_state
            )

            features = build_correlation_features(
                sample,
                include_coordinates=include_coordinates
            )

            # Remove unavailable and constant features
            features = features.dropna(
                axis=1, how="all"
            )

            variable_columns = [
                col for col in features.columns
                if features[col].nunique(dropna=True) > 1
            ]

            features = features[variable_columns]

            if features.shape[1] < 2:
                print(
                    f"Skipping {period}: insufficient variables"
                )
                continue

            # Pearson: linear association
            pearson = features.corr(
                method="pearson",
                min_periods=100
            )

            # Spearman: monotonic/rank association
            spearman = features.corr(
                method="spearman",
                min_periods=100
            )

            # ------------------------------------
            # 4. Save correlation matrices
            # ------------------------------------

            output_path = (
                Path(correlation_root)
                / sensor_name
                / platform_name
                / period
            )

            output_path.mkdir(
                parents=True,
                exist_ok=True
            )

            pearson.to_csv(
                output_path / "pearson.csv"
            )

            spearman.to_csv(
                output_path / "spearman.csv"
            )

            results[
                (sensor_name, platform_name)
            ][period] = {
                "pearson": pearson,
                "spearman": spearman,
                "sample_size": len(sample),
                "total_records": len(subset),
                "features": features.columns.tolist()
            }

            print(
                f"  {period}: "
                f"{len(sample):,} sampled from "
                f"{len(subset):,} records"
            )

    return results


# ==================================================
# Shared categorical definitions
# ==================================================

SENSOR_CATEGORIES = ["MODIS", "VIIRS"]

PLATFORM_CATEGORIES = [
    "Terra",
    "Aqua",
    "Suomi NPP",
    "NOAA-20",
    "NOAA-21"
]

SATELLITE_CATEGORIES = [
    "Aqua",
    "Terra",
    "N",
    "N20",
    "N21"
]

CONFIDENCE_CATEGORIES = [
    "low",
    "nominal",
    "high"
]

VIIRS_CONFIDENCE_CATEGORIES = ["l", "n", "h"]

DAYNIGHT_CATEGORIES = ["DAY", "NIGHT"]

FIRE_TYPE_CATEGORIES = [
    "presumed_vegetation_fire",
    "active_volcano",
    "other_static_land_source",
    "offshore_detection"
]

PRODUCT_CATEGORIES = [
    "MODIS FIRMS archive",
    "VIIRS FIRMS 375m archive"
]

MID_IR_CHANNELS = [
    "MODIS_B21_B22",
    "VIIRS_I4"
]

THERMAL_CHANNELS = [
    "MODIS_B31",
    "VIIRS_I5"
]


# ==================================================
# 1. Numeric conversion without intermediate frames
# ==================================================

def _numeric_array(
    df: pd.DataFrame,
    column: str,
    dtype=np.float32
) -> np.ndarray:

    if column not in df.columns:
        return np.full(len(df), np.nan, dtype=dtype)

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return values.to_numpy(
        dtype=dtype,
        na_value=np.nan,
        copy=False
    )


def _first_numeric_array(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    dtype=np.float32
) -> np.ndarray:

    for column in columns:

        if column in df.columns:
            return _numeric_array(
                df, column, dtype
            )

    return np.full(len(df), np.nan, dtype=dtype)


def _categorical(
    codes: np.ndarray,
    categories: list,
    ordered: bool = False
) -> pd.Categorical:

    return pd.Categorical.from_codes(
        codes,
        categories=categories,
        ordered=ordered
    )


# ==================================================
# 2. Fast datetime conversion
# ==================================================

def _datetime_utc(df: pd.DataFrame) -> pd.Series:

    dates = pd.to_datetime(
        df["acq_date"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    raw_time = pd.to_numeric(
        df["acq_time"],
        errors="coerce"
    ).to_numpy(
        dtype=np.float64,
        na_value=np.nan
    )

    # Validate HHMM
    whole = (
        np.isfinite(raw_time)
        & (raw_time == np.floor(raw_time))
        & (raw_time >= 0)
        & (raw_time <= 2359)
    )

    hhmm = np.zeros(len(df), dtype=np.int32)
    hhmm[whole] = raw_time[whole].astype(np.int32)

    hours = hhmm // 100
    minutes = hhmm % 100

    valid = (
        whole
        & (hours < 24)
        & (minutes < 60)
    )

    minute_of_day = np.full(
        len(df), np.nan, dtype=np.float64
    )

    minute_of_day[valid] = (
        hours[valid] * 60 + minutes[valid]
    )

    delta = pd.to_timedelta(
        minute_of_day,
        unit="m"
    )

    result = dates + delta

    return result.dt.tz_localize("UTC")


# ==================================================
# 3. Main harmonization function
# ==================================================

def _harmonize_one(
    df: pd.DataFrame,
    source_key: str,
    source_file_categories: list[str] | None = None,
    source_file_value: str | None = None,
    extra_spatial_features: bool = False
) -> pd.DataFrame:

    source_key = source_key.upper()

    source_sensor = {
        "MODIS": "MODIS",
        "SV": "VIIRS",
        "J1V": "VIIRS",
        "J2V": "VIIRS"
    }

    if source_key not in source_sensor:
        raise ValueError(f"Unknown source: {source_key}")

    sensor = source_sensor[source_key]
    n = len(df)

    # ----------------------------------------------
    # Required fields
    # ----------------------------------------------

    required = [
        "latitude",
        "longitude",
        "acq_date",
        "acq_time",
        "satellite"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{source_key}: missing columns {missing}"
        )

    # ----------------------------------------------
    # Sensor codes
    # ----------------------------------------------

    sensor_code = 0 if sensor == "MODIS" else 1

    sensor_codes = np.full(
        n, sensor_code, dtype=np.int8
    )

    sensor_cat = _categorical(
        sensor_codes,
        SENSOR_CATEGORIES
    )

    # ----------------------------------------------
    # Satellite and platform
    # ----------------------------------------------

    satellite_cat = pd.Categorical(
        df["satellite"],
        categories=SATELLITE_CATEGORIES
    )

    satellite_codes = satellite_cat.codes

    # Aqua -> 1, Terra -> 0,
    # N -> 2, N20 -> 3, N21 -> 4
    platform_lookup = np.array(
        [1, 0, 2, 3, 4],
        dtype=np.int8
    )

    platform_codes = np.full(
        n, -1, dtype=np.int8
    )

    valid_satellite = satellite_codes >= 0

    platform_codes[valid_satellite] = (
        platform_lookup[
            satellite_codes[valid_satellite]
        ]
    )

    platform_cat = _categorical(
        platform_codes,
        PLATFORM_CATEGORIES
    )

    # ----------------------------------------------
    # Confidence
    # ----------------------------------------------

    confidence_numeric = np.full(
        n, np.nan, dtype=np.float32
    )

    confidence_label_codes = np.full(
        n, -1, dtype=np.int8
    )

    confidence_codes = np.full(
        n, -1, dtype=np.int8
    )

    if sensor == "MODIS":

        confidence_numeric = _numeric_array(
            df, "confidence"
        )

        valid_conf = (
            np.isfinite(confidence_numeric)
            & (confidence_numeric >= 0)
            & (confidence_numeric <= 100)
        )

        confidence_codes[
            valid_conf & (confidence_numeric < 30)
        ] = 0

        confidence_codes[
            valid_conf
            & (confidence_numeric >= 30)
            & (confidence_numeric < 80)
        ] = 1

        confidence_codes[
            valid_conf & (confidence_numeric >= 80)
        ] = 2

    else:

        raw_labels = pd.Categorical(
            df["confidence"],
            categories=VIIRS_CONFIDENCE_CATEGORIES
        )

        confidence_codes = raw_labels.codes.astype(
            np.int8, copy=False
        )

        confidence_label_codes = confidence_codes.copy()

    confidence_rank = confidence_codes.copy()

    confidence_class = _categorical(
        confidence_codes,
        CONFIDENCE_CATEGORIES,
        ordered=True
    )

    confidence_raw_label = _categorical(
        confidence_label_codes,
        VIIRS_CONFIDENCE_CATEGORIES
    )

    # ----------------------------------------------
    # Brightness measurements
    # ----------------------------------------------

    if sensor == "MODIS":

        mid_ir = _first_numeric_array(
            df, ("brightness",)
        )

        thermal = _first_numeric_array(
            df, ("bright_t31",)
        )

        mid_channel = "MODIS_B21_B22"
        thermal_channel = "MODIS_B31"

    else:

        mid_ir = _first_numeric_array(
            df,
            ("bright_ti4", "bright_t4", "brightness")
        )

        thermal = _first_numeric_array(
            df,
            (
                "bright_ti5",
                "bright_t5",
                "bright_t31",
                "brightness_2"
            )
        )

        mid_channel = "VIIRS_I4"
        thermal_channel = "VIIRS_I5"

    bt_contrast = np.subtract(
        mid_ir,
        thermal
    )

    # ----------------------------------------------
    # Geographic information
    # Keep coordinates in float64
    # ----------------------------------------------

    latitude = _numeric_array(
        df, "latitude", np.float64
    )

    longitude = _numeric_array(
        df, "longitude", np.float64
    )

    location_valid = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= -90)
        & (latitude <= 90)
        & (longitude >= -180)
        & (longitude <= 180)
    )

    # ----------------------------------------------
    # Spatial footprint
    # ----------------------------------------------

    scan = _numeric_array(df, "scan")
    track = _numeric_array(df, "track")

    valid_footprint = (
        np.isfinite(scan)
        & np.isfinite(track)
        & (scan > 0)
        & (track > 0)
    )

    pixel_area = np.full(
        n, np.nan, dtype=np.float32
    )

    np.multiply(
        scan,
        track,
        out=pixel_area,
        where=valid_footprint
    )

    # Optional derived features
    if extra_spatial_features:

        equivalent_width = np.sqrt(pixel_area)

        aspect_ratio = np.full(
            n, np.nan, dtype=np.float32
        )

        np.divide(
            scan,
            track,
            out=aspect_ratio,
            where=valid_footprint
        )

    # ----------------------------------------------
    # Day / night
    # ----------------------------------------------

    raw_dn = pd.Categorical(
        df["daynight"],
        categories=["D", "DAY", "N", "NIGHT"]
    )

    dn_codes = raw_dn.codes

    daynight_codes = np.full(
        n, -1, dtype=np.int8
    )

    daynight_codes[
        np.isin(dn_codes, [0, 1])
    ] = 0

    daynight_codes[
        np.isin(dn_codes, [2, 3])
    ] = 1

    daynight_cat = _categorical(
        daynight_codes,
        DAYNIGHT_CATEGORIES
    )

    # ----------------------------------------------
    # Fire type
    # ----------------------------------------------

    fire_type_code = _numeric_array(
        df, "type"
    )

    valid_type = (
        np.isfinite(fire_type_code)
        & (fire_type_code >= 0)
        & (fire_type_code <= 3)
        & (fire_type_code == np.floor(fire_type_code))
    )

    type_codes = np.full(
        n, -1, dtype=np.int8
    )

    type_codes[valid_type] = (
        fire_type_code[valid_type].astype(np.int8)
    )

    fire_type_cat = _categorical(
        type_codes,
        FIRE_TYPE_CATEGORIES
    )

    # ----------------------------------------------
    # Version
    # ----------------------------------------------

    version = _numeric_array(
        df, "version"
    )

    # ----------------------------------------------
    # Source file
    # ----------------------------------------------

    if source_file_value is not None:

        categories = source_file_categories or [
            source_file_value
        ]

        source_code = categories.index(
            source_file_value
        )

        source_file_cat = _categorical(
            np.full(n, source_code, dtype=np.int32),
            categories
        )

    elif "source_file" in df.columns:

        categories = source_file_categories

        if categories is None:
            categories = sorted(
                df["source_file"].dropna().unique().tolist()
            )

        source_file_cat = pd.Categorical(
            df["source_file"],
            categories=categories
        )

    else:

        source_file_cat = _categorical(
            np.full(n, -1, dtype=np.int8),
            source_file_categories or []
        )

    # ----------------------------------------------
    # Source product
    # ----------------------------------------------

    product_code = (
        0 if sensor == "MODIS" else 1
    )

    product_cat = _categorical(
        np.full(n, product_code, dtype=np.int8),
        PRODUCT_CATEGORIES
    )

    mid_channel_cat = _categorical(
        np.full(
            n,
            MID_IR_CHANNELS.index(mid_channel),
            dtype=np.int8
        ),
        MID_IR_CHANNELS
    )

    thermal_channel_cat = _categorical(
        np.full(
            n,
            THERMAL_CHANNELS.index(thermal_channel),
            dtype=np.int8
        ),
        THERMAL_CHANNELS
    )

    # ----------------------------------------------
    # Construct only the final DataFrame
    # ----------------------------------------------

    data = {
        "sensor": sensor_cat,
        "instrument": sensor_cat,
        "source_product": product_cat,
        "source_file": source_file_cat,
        "version": version,

        "satellite_code": satellite_cat,
        "platform": platform_cat,

        "latitude": latitude,
        "longitude": longitude,
        "location_valid": location_valid,
        "acq_datetime_utc": _datetime_utc(df),

        "confidence_raw_numeric": confidence_numeric,
        "confidence_raw_label": confidence_raw_label,
        "confidence_class": confidence_class,
        "confidence_rank": confidence_rank,

        "bt_mid_ir_k": mid_ir,
        "bt_thermal_k": thermal,
        "bt_contrast_k": bt_contrast,

        "bt_mid_ir_channel": mid_channel_cat,
        "bt_thermal_channel": thermal_channel_cat,

        "scan_km": scan,
        "track_km": track,
        "pixel_area_km2": pixel_area,

        "frp_mw": _numeric_array(df, "frp"),

        "daynight": daynight_cat,

        "fire_type_code": fire_type_code,
        "fire_type": fire_type_cat,
    }

    if extra_spatial_features:
        data["pixel_equivalent_width_km"] = equivalent_width
        data["pixel_aspect_ratio"] = aspect_ratio

    return pd.DataFrame(data)


# ==================================================
# 4. Combine all sources
# ==================================================

def combine_firms_datasets(
    datasets: dict[str, pd.DataFrame],
    extra_spatial_features: bool = False
) -> pd.DataFrame:

    # Collect source filenames once so all categorical
    # columns have identical categories before concatenation.
    source_files = set()

    for df in datasets.values():

        if "source_file" in df.columns:

            source_files.update(
                pd.unique(
                    df["source_file"].dropna()
                ).tolist()
            )

    source_file_categories = sorted(
        source_files,
        key=str
    )

    parts = []

    for source_key, df in datasets.items():

        if df.empty:
            continue

        part = _harmonize_one(
            df,
            source_key=source_key,
            source_file_categories=source_file_categories,
            extra_spatial_features=extra_spatial_features
        )

        parts.append(part)

        print(
            f"{source_key}: {len(part):,} records harmonized"
        )

    if not parts:
        return pd.DataFrame()

    # No sorting: sorting 25 million rows is expensive.
    combined = pd.concat(
        parts,
        ignore_index=True,
        copy=False
    )

    return combined

# --------------------- Memory Helper ---------------------

def show_memory_usage(df, name="DataFrame"):

    memory_mb = (
        df.memory_usage(deep=True).sum()
        / 1024**2
    )

    print(
        f"{name}: {memory_mb:,.2f} MB "
        f"({memory_mb / 1024:,.2f} GB)"
    )

    print(
        df.dtypes.value_counts()
    )


def harmonize_csvs_to_parquet(
    files_by_source: dict,
    output_dir: str = "data/harmonized",
    chunksize: int = 250_000
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_columns = {
        "latitude",
        "longitude",
        "brightness",
        "bright_t31",
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "version",
        "frp",
        "daynight",
        "type"
    }

    total_written = 0

    for source_key, files in files_by_source.items():

        source_folder = (
            output_dir / f"source={source_key}"
        )

        source_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for csv_file in sorted(map(Path, files)):

            print(f"Processing: {csv_file.name}")

            chunk_reader = pd.read_csv(
                csv_file,
                usecols=lambda col: col in raw_columns,
                chunksize=chunksize,
                low_memory=False
            )

            for part_number, chunk in enumerate(chunk_reader):

                harmonized = _harmonize_one(
                    chunk,
                    source_key=source_key,
                    source_file_categories=[csv_file.name],
                    source_file_value=csv_file.name
                )

                destination = (
                    source_folder
                    / f"{csv_file.stem}_{part_number:05d}.parquet"
                )

                harmonized.to_parquet(
                    destination,
                    engine="pyarrow",
                    compression="zstd",
                    index=False
                )

                total_written += len(harmonized)

                print(
                    f"  Saved {len(harmonized):,} records"
                )

                del chunk, harmonized

            del chunk_reader
            gc.collect()

    print(
        f"\nTotal records written: {total_written:,}"
    )

    return output_dir



#--------------- BEFORE CLUSTRING POINTS WE SHOULD REMOVE DUPLICATED POINTS ---------------
def remove_duplicate_records(df: pd.DataFrame) -> pd.DataFrame:

    keys = [
        "sensor",
        "satellite_code",
        "acq_datetime_utc",
        "latitude",
        "longitude",
        "version",
        "bt_mid_ir_k",
        "bt_thermal_k",
        "frp_mw",
        "confidence_raw",
        "daynight",
    ]

    keys = [col for col in keys if col in df.columns]

    return (
        df.drop_duplicates(subset=keys)
        .reset_index(drop=True)
    )

# ---------------------- Fire Event Clustring ----------------------

def assign_estimated_fire_events(
    df: pd.DataFrame,
    radius_km: float = 1.5,
    max_time_gap_hours: float = 12,
    batch_size: int = 2000,
) -> pd.DataFrame:

    if radius_km <= 0 or max_time_gap_hours <= 0:
        raise ValueError("Clustering thresholds must be positive")

    result = df.reset_index(drop=True).copy()

    result["fire_event_id"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    # Only records with valid coordinates and timestamps
    valid = (
        result["latitude"].between(-90, 90)
        & result["longitude"].between(-180, 180)
        & result["acq_datetime_utc"].notna()
    )

    positions = np.flatnonzero(valid.to_numpy())

    n = len(positions)

    if n == 0:
        return result

    # Geographic coordinates in radians
    coordinates = np.deg2rad(
        result.loc[
            positions, ["latitude", "longitude"]
        ].to_numpy(dtype=float)
    )

    # Acquisition timestamps in nanoseconds
    timestamps = (
        result.loc[positions, "acq_datetime_utc"]
        .astype("int64")
        .to_numpy()
    )

    # Build spatial neighbor index
    earth_radius_km = 6371.0088

    tree = BallTree(
        coordinates,
        metric="haversine"
    )

    spatial_radius = radius_km / earth_radius_km

    temporal_radius = pd.Timedelta(
        hours=max_time_gap_hours
    ).value

    edge_rows = []
    edge_cols = []

    # Search in batches to limit temporary memory usage
    for start in range(0, n, batch_size):

        stop = min(start + batch_size, n)

        neighborhoods = tree.query_radius(
            coordinates[start:stop],
            r=spatial_radius
        )

        for offset, neighbors in enumerate(neighborhoods):

            i = start + offset

            # Avoid duplicate edges and self-connections
            neighbors = neighbors[neighbors > i]

            if len(neighbors) == 0:
                continue

            # Apply temporal condition
            time_difference = np.abs(
                timestamps[neighbors] - timestamps[i]
            )

            matched = neighbors[
                time_difference <= temporal_radius
            ]

            if len(matched) == 0:
                continue

            edge_rows.extend([i] * len(matched))
            edge_cols.extend(matched.tolist())

    # Construct undirected graph
    adjacency = coo_matrix(
        (
            np.ones(len(edge_rows), dtype=np.uint8),
            (edge_rows, edge_cols)
        ),
        shape=(n, n)
    ).tocsr()

    adjacency = adjacency + adjacency.T

    # Identify connected components
    n_components, labels = connected_components(
        adjacency,
        directed=False
    )

    # Order event IDs by their first observation
    first_seen = np.full(
        n_components,
        np.iinfo(np.int64).max,
        dtype=np.int64
    )

    np.minimum.at(
        first_seen,
        labels,
        timestamps
    )

    order = np.argsort(first_seen, kind="stable")

    event_ids = np.empty(n_components, dtype=np.int64)
    event_ids[order] = np.arange(1, n_components + 1)

    result.loc[positions, "fire_event_id"] = (
        event_ids[labels]
    )

    return result





# ---------------------- Spatial Grid ----------------------

def add_spatial_grid(
    df: pd.DataFrame,
    grid_size_km: float = 5.0,
    lat_col: str = "latitude",
    lon_col: str = "longitude"
) -> pd.DataFrame:

    if grid_size_km <= 0:
        raise ValueError("grid_size_km must be positive")

    result = df.copy()

    cell_size_m = grid_size_km * 1000

    # WGS84 geographic coordinates -> global equal-area grid
    to_grid = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:6933",
        always_xy=True
    )

    # Grid coordinates -> WGS84 for cell-center display
    to_wgs84 = Transformer.from_crs(
        "EPSG:6933",
        "EPSG:4326",
        always_xy=True
    )

    latitude = pd.to_numeric(
        result[lat_col], errors="coerce"
    )

    longitude = pd.to_numeric(
        result[lon_col], errors="coerce"
    )

    # EPSG:6933 coverage is between 86°S and 86°N
    valid = (
        latitude.between(-86, 86)
        & longitude.between(-180, 180)
    )

    result["grid_col"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    result["grid_row"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    result["grid_id"] = pd.Series(
        pd.NA, index=result.index, dtype="string"
    )

    result["grid_center_lat"] = np.nan
    result["grid_center_lon"] = np.nan

    positions = np.flatnonzero(valid.to_numpy())

    if len(positions) == 0:
        return result

    x, y = to_grid.transform(
        longitude.iloc[positions].to_numpy(),
        latitude.iloc[positions].to_numpy()
    )

    valid_projection = (
        np.isfinite(x) & np.isfinite(y)
    )

    positions = positions[valid_projection]

    x = np.asarray(x)[valid_projection]
    y = np.asarray(y)[valid_projection]

    # Calculate cell indices
    grid_col = np.floor(
        x / cell_size_m
    ).astype("int64")

    grid_row = np.floor(
        y / cell_size_m
    ).astype("int64")

    result.iloc[
        positions,
        result.columns.get_loc("grid_col")
    ] = grid_col

    result.iloc[
        positions,
        result.columns.get_loc("grid_row")
    ] = grid_row

    # Stable cell identifier for a given grid resolution
    grid_ids = [
        f"{grid_size_km:g}km:{col}:{row}"
        for col, row in zip(grid_col, grid_row)
    ]

    result.iloc[
        positions,
        result.columns.get_loc("grid_id")
    ] = grid_ids

    # Calculate the center of each cell
    center_x = (grid_col + 0.5) * cell_size_m
    center_y = (grid_row + 0.5) * cell_size_m

    center_lon, center_lat = to_wgs84.transform(
        center_x, center_y
    )

    result.iloc[
        positions,
        result.columns.get_loc("grid_center_lat")
    ] = center_lat

    result.iloc[
        positions,
        result.columns.get_loc("grid_center_lon")
    ] = center_lon

    return result



#  ------------------- Aggrating Function -------------------


def aggregate_detection_grid(
    df: pd.DataFrame,
    frequency: str = "monthly"
) -> pd.DataFrame:

    if frequency not in {"daily", "monthly"}:
        raise ValueError(
            "frequency must be 'daily' or 'monthly'"
        )

    data = df.dropna(
        subset=[
            "grid_id",
            "acq_datetime_utc",
            "sensor"
        ]
    ).copy()

    if frequency == "daily":
        data["period"] = (
            data["acq_datetime_utc"]
            .dt.strftime("%Y-%m-%d")
        )

    else:
        data["period"] = (
            data["acq_datetime_utc"]
            .dt.strftime("%Y-%m")
        )

    group_columns = [
        "period",
        "grid_id",
        "grid_col",
        "grid_row",
        "sensor"
    ]

    result = (
        data.groupby(
            group_columns,
            observed=True,
            dropna=False
        )
        .agg(
            detection_count=(
                "sensor", "size"
            ),
            platform_count=(
                "platform", "nunique"
            ),
            mean_frp_mw=(
                "frp_mw", "mean"
            ),
            max_frp_mw=(
                "frp_mw", "max"
            ),
            high_confidence_count=(
                "confidence_class",
                lambda s: (s == "high").sum()
            ),
            nominal_confidence_count=(
                "confidence_class",
                lambda s: (s == "nominal").sum()
            ),
            low_confidence_count=(
                "confidence_class",
                lambda s: (s == "low").sum()
            ),
            day_detection_count=(
                "daynight",
                lambda s: (s == "DAY").sum()
            ),
            night_detection_count=(
                "daynight",
                lambda s: (s == "NIGHT").sum()
            ),
        )
        .reset_index()
    )

    # Recover geographic center coordinates
    cell_centers = (
        data[
            [
                "grid_id",
                "grid_center_lat",
                "grid_center_lon"
            ]
        ]
        .drop_duplicates("grid_id")
    )

    result = result.merge(
        cell_centers,
        on="grid_id",
        how="left",
        validate="many_to_one"
    )

    return result
