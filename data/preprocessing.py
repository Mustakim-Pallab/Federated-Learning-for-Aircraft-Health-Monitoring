from __future__ import annotations

import os
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CMAPSS_COLUMNS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

DEFAULT_SENSORS = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]


@dataclass
class WindowedData:
    x: np.ndarray
    rul: np.ndarray
    fault: np.ndarray
    unit: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])


def maybe_download_cmapss(url: str, data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    train_file = data_dir / "train_FD001.txt"
    if train_file.exists():
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "cmapss.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(data_dir)


def _read_txt(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing C-MAPSS file: {path}")
    return pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)


def load_cmapss(
    data_dir: str | Path,
    subset: str = "FD001",
    rul_cap: int = 125,
    fault_threshold: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = _read_txt(data_dir / f"train_{subset}.txt")
    test = _read_txt(data_dir / f"test_{subset}.txt")
    rul_path = data_dir / f"RUL_{subset}.txt"
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing C-MAPSS test RUL file: {rul_path}")

    final_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["final_rul"])
    max_train_cycles = train.groupby("unit")["cycle"].max()
    train["rul_raw"] = train.apply(
        lambda row: max_train_cycles.loc[row["unit"]] - row["cycle"], axis=1
    )

    max_test_cycles = test.groupby("unit")["cycle"].max()
    final_rul_by_unit = dict(zip(sorted(test["unit"].unique()), final_rul["final_rul"]))
    test["rul_raw"] = test.apply(
        lambda row: max_test_cycles.loc[row["unit"]]
        - row["cycle"]
        + final_rul_by_unit[row["unit"]],
        axis=1,
    )

    for frame in (train, test):
        frame["rul"] = frame["rul_raw"].clip(upper=rul_cap).astype(np.float32)
        frame["fault"] = (frame["rul_raw"] <= fault_threshold).astype(np.float32)

    return train, test


def split_units(
    units: list[int],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = list(units)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * validation_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def partition_units(
    units: list[int],
    num_clients: int,
    seed: int,
    strategy: str = "iid",
) -> dict[int, list[int]]:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    rng = random.Random(seed)
    shuffled = list(units)
    rng.shuffle(shuffled)

    if strategy == "iid":
        clients = {client_id: [] for client_id in range(num_clients)}
        for index, unit in enumerate(shuffled):
            clients[index % num_clients].append(unit)
        return clients

    if strategy == "ordered":
        chunks = np.array_split(sorted(units), num_clients)
        return {client_id: [int(x) for x in chunk] for client_id, chunk in enumerate(chunks)}

    raise ValueError(f"Unknown partition strategy: {strategy}")


def fit_scaler(frame: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(frame[feature_cols])
    return scaler


def apply_scaler(frame: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    scaled = frame.copy()
    scaled[feature_cols] = scaler.transform(scaled[feature_cols])
    return scaled


def make_windows(
    frame: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    stride: int,
) -> WindowedData:
    xs: list[np.ndarray] = []
    ruls: list[float] = []
    faults: list[float] = []
    units: list[int] = []

    for unit, unit_df in frame.groupby("unit"):
        unit_df = unit_df.sort_values("cycle")
        features = unit_df[feature_cols].to_numpy(dtype=np.float32)
        rul = unit_df["rul"].to_numpy(dtype=np.float32)
        fault = unit_df["fault"].to_numpy(dtype=np.float32)

        if len(unit_df) < window_size:
            continue

        for start in range(0, len(unit_df) - window_size + 1, stride):
            end = start + window_size
            xs.append(features[start:end])
            ruls.append(float(rul[end - 1]))
            faults.append(float(fault[end - 1]))
            units.append(int(unit))

    if not xs:
        return WindowedData(
            x=np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            rul=np.empty((0,), dtype=np.float32),
            fault=np.empty((0,), dtype=np.float32),
            unit=np.empty((0,), dtype=np.int64),
        )

    return WindowedData(
        x=np.stack(xs).astype(np.float32),
        rul=np.asarray(ruls, dtype=np.float32),
        fault=np.asarray(faults, dtype=np.float32),
        unit=np.asarray(units, dtype=np.int64),
    )


def subset_units(frame: pd.DataFrame, units: list[int]) -> pd.DataFrame:
    return frame[frame["unit"].isin(units)].copy()


def prepare_cmapss_clients(config: dict) -> dict:
    ds_cfg = config["dataset"]
    fl_cfg = config["federated"]

    if ds_cfg.get("download", False):
        maybe_download_cmapss(ds_cfg["url"], ds_cfg["data_dir"])

    feature_cols = ds_cfg.get("sensors") or DEFAULT_SENSORS
    train_df, test_df = load_cmapss(
        data_dir=ds_cfg["data_dir"],
        subset=ds_cfg["name"],
        rul_cap=ds_cfg["rul_cap"],
        fault_threshold=ds_cfg["fault_threshold"],
    )

    train_units, val_units = split_units(
        sorted(train_df["unit"].unique().tolist()),
        validation_fraction=ds_cfg["validation_engine_fraction"],
        seed=config["seed"],
    )

    train_only = subset_units(train_df, train_units)
    val_only = subset_units(train_df, val_units)
    scaler = fit_scaler(train_only, feature_cols)
    train_only = apply_scaler(train_only, feature_cols, scaler)
    val_only = apply_scaler(val_only, feature_cols, scaler)
    test_df = apply_scaler(test_df, feature_cols, scaler)

    client_units = partition_units(
        train_units,
        num_clients=fl_cfg["num_clients"],
        seed=config["seed"],
        strategy=fl_cfg.get("partition_strategy", "iid"),
    )

    clients = {}
    for client_id, units in client_units.items():
        clients[client_id] = {
            "train": make_windows(
                subset_units(train_only, units),
                feature_cols,
                ds_cfg["window_size"],
                ds_cfg["stride"],
            ),
            "units": units,
        }

    return {
        "clients": clients,
        "validation": make_windows(val_only, feature_cols, ds_cfg["window_size"], ds_cfg["stride"]),
        "test": make_windows(test_df, feature_cols, ds_cfg["window_size"], ds_cfg["stride"]),
        "feature_cols": feature_cols,
        "input_size": len(feature_cols),
        "scaler": scaler,
    }
