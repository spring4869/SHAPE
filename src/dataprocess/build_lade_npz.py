import argparse
import os
import pickle
from datetime import datetime, timedelta
from math import atan2, cos, pi, sin

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


HOLIDAYS = [
    ("2022-01-01", "2022-01-03"),
    ("2022-01-31", "2022-02-06"),
    ("2022-04-03", "2022-04-05"),
    ("2022-04-30", "2022-05-04"),
    ("2022-06-03", "2022-06-05"),
    ("2022-09-10", "2022-09-12"),
    ("2022-10-01", "2022-10-07"),
]

WORKDAYS = [
    "2022-01-29",
    "2022-01-30",
    "2022-04-02",
    "2022-04-24",
    "2022-05-07",
    "2022-10-08",
    "2022-10-09",
]


def build_calendar_sets():
    holiday_ranges = []
    for start, end in HOLIDAYS:
        d1 = datetime.strptime(start, "%Y-%m-%d")
        d2 = datetime.strptime(end, "%Y-%m-%d")
        while d1 <= d2:
            holiday_ranges.append(d1.strftime("%Y-%m-%d"))
            d1 += timedelta(days=1)
    return set(holiday_ranges), set(WORKDAYS)


HOLIDAY_SET, WORKDAY_SET = build_calendar_sets()


def encode_postman_id(postman_id, id_map):
    if postman_id not in id_map:
        id_map[postman_id] = len(id_map)
    return id_map[postman_id]


def encode_time(dep_time):
    seconds_in_day = dep_time.hour * 3600 + dep_time.minute * 60 + dep_time.second
    weekday = dep_time.weekday()
    dep_date_str = dep_time.strftime("%Y-%m-%d")
    return [
        sin(2 * pi * seconds_in_day / 86400),
        cos(2 * pi * seconds_in_day / 86400),
        sin(2 * pi * weekday / 7),
        cos(2 * pi * weekday / 7),
        float(weekday >= 5),
        sin(2 * pi * dep_time.day / 31),
        cos(2 * pi * dep_time.day / 31),
        sin(2 * pi * dep_time.month / 12),
        cos(2 * pi * dep_time.month / 12),
        float((dep_date_str in HOLIDAY_SET) or (dep_date_str in WORKDAY_SET)),
    ]


def extract_features_from_df(traj_df):
    features = []
    cumulative_dist = 0.0
    lats = traj_df["lat"].values
    lngs = traj_df["lng"].values
    prev_bearing = None

    for i in range(1, len(lats)):
        lat1, lng1 = lats[i - 1], lngs[i - 1]
        lat2, lng2 = lats[i], lngs[i]
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        bearing = atan2(dlng, dlat)
        segment_dist = np.sqrt(dlat**2 + dlng**2)
        cumulative_dist += segment_dist

        if prev_bearing is not None:
            bearing_change = bearing - prev_bearing
            if bearing_change > pi:
                bearing_change -= 2 * pi
            if bearing_change < -pi:
                bearing_change += 2 * pi
        else:
            bearing_change = 0.0
        curvature = abs(bearing_change) / (segment_dist + 1e-9)
        prev_bearing = bearing

        features.append([dlat, dlng, sin(bearing), cos(bearing), cumulative_dist, bearing_change, curvature])
    return np.array(features), cumulative_dist


def build_dataset(X_raw, s_raw, e_raw, d_raw, t_raw, eta_raw, scaler_ri, scaler_latlng, scaler_len, max_len):
    feature_dim = 7
    X_norm = [scaler_ri.transform(feats) if feats.shape[0] > 0 else np.empty((0, feature_dim)) for feats in X_raw]

    latlng_all = scaler_latlng.transform(np.array(s_raw + e_raw))
    s_scaled, e_scaled = [], []
    for i in range(len(s_raw)):
        s_scaled.append(np.pad(latlng_all[2 * i], (0, feature_dim - 2)))
        e_scaled.append(np.pad(latlng_all[2 * i + 1], (0, feature_dim - 2)))

    d_scaled = scaler_len.transform(np.array(d_raw).reshape(-1, 1)).reshape(-1)
    d_tokens = [np.pad([d], (0, feature_dim - 1)) for d in d_scaled]

    sep = np.ones((1, feature_dim))
    pad = np.zeros((1, feature_dim))
    X_final, mask_final = [], []
    for feats, s, e, d_token in zip(X_norm, s_scaled, e_scaled, d_tokens):
        seq = np.vstack([s.reshape(1, -1), feats, e.reshape(1, -1), d_token.reshape(1, -1), sep])
        mask = np.ones(len(seq))
        if len(seq) < max_len:
            pad_len = max_len - len(seq)
            seq = np.vstack([seq, np.repeat(pad, pad_len, axis=0)])
            mask = np.concatenate([mask, np.zeros(pad_len)])
        X_final.append(seq)
        mask_final.append(mask)

    return np.array(X_final), np.array(mask_final), np.array(t_raw), np.array(eta_raw)


def main():
    parser = argparse.ArgumentParser(description="Build processed npz data from trajectory_segment_*.pkl files.")
    parser.add_argument("--traj-dir", required=True, help="Directory containing trajectory_segment_*.pkl files")
    parser.add_argument("--output-dir", required=True, help="Directory to save processed npz, scalers, and postman map")
    parser.add_argument("--max-len", type=int, default=55)
    parser.add_argument("--min-len", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X_data, traj_raw_data, start_tokens, end_tokens = [], [], [], []
    distances, time_encoding, etas = [], [], []
    postman_mapping = {}

    file_list = sorted(os.listdir(args.traj_dir))
    for file in tqdm(file_list, desc="Processing Trajectories"):
        if not file.endswith(".pkl"):
            continue
        try:
            with open(os.path.join(args.traj_dir, file), "rb") as f:
                traj_df = pickle.load(f)
            if not isinstance(traj_df, pd.DataFrame) or traj_df.empty:
                continue
        except Exception:
            continue

        if not (args.min_len <= len(traj_df) <= args.max_len):
            continue

        feats, total_distance = extract_features_from_df(traj_df)
        if len(feats) + 5 > args.max_len:
            continue

        start_row = traj_df.iloc[0]
        end_row = traj_df.iloc[-1]
        dep_time = start_row["gps_time"]
        arr_time = end_row["gps_time"]
        if not isinstance(dep_time, datetime) or not isinstance(arr_time, datetime):
            continue

        duration = (arr_time - dep_time).total_seconds()
        if not (0 <= duration <= 36000):
            continue

        X_data.append(feats)
        traj_raw_data.append(traj_df[["lat", "lng"]].values)
        start_tokens.append([float(start_row["lat"]), float(start_row["lng"])])
        end_tokens.append([float(end_row["lat"]), float(end_row["lng"])])
        distances.append(total_distance)
        etas.append(duration)

        encoded_id = encode_postman_id(start_row["postman_id"], postman_mapping)
        time_encoding.append(encode_time(dep_time) + [encoded_id])

    split_kwargs = {"test_size": 0.4, "random_state": args.seed}
    X_train_raw, X_temp_raw, traj_train_raw, traj_temp_raw, s_train, s_temp, e_train, e_temp, d_train, d_temp, t_train, t_temp, eta_train, eta_temp = train_test_split(
        X_data, traj_raw_data, start_tokens, end_tokens, distances, time_encoding, etas, **split_kwargs
    )
    X_val_raw, X_test_raw, traj_val_raw, traj_test_raw, s_val, s_test, e_val, e_test, d_val, d_test, t_val, t_test, eta_val, eta_test = train_test_split(
        X_temp_raw, traj_temp_raw, s_temp, e_temp, d_temp, t_temp, eta_temp, test_size=0.5, random_state=args.seed
    )

    scaler_ri = StandardScaler().fit(np.concatenate(X_train_raw, axis=0))
    scaler_latlng = StandardScaler().fit(np.array(s_train + e_train))
    scaler_len = StandardScaler().fit(np.array(d_train).reshape(-1, 1))

    with open(os.path.join(args.output_dir, "postman_id_map.pkl"), "wb") as f:
        pickle.dump(postman_mapping, f)
    with open(os.path.join(args.output_dir, "scaler_ri.pkl"), "wb") as f:
        pickle.dump(scaler_ri, f)
    with open(os.path.join(args.output_dir, "scaler_latlng.pkl"), "wb") as f:
        pickle.dump(scaler_latlng, f)
    with open(os.path.join(args.output_dir, "scaler_len.pkl"), "wb") as f:
        pickle.dump(scaler_len, f)

    X_train, mask_train, time_encoding_train, etas_train = build_dataset(
        X_train_raw, s_train, e_train, d_train, t_train, eta_train, scaler_ri, scaler_latlng, scaler_len, args.max_len
    )
    X_val, mask_val, time_encoding_val, etas_val = build_dataset(
        X_val_raw, s_val, e_val, d_val, t_val, eta_val, scaler_ri, scaler_latlng, scaler_len, args.max_len
    )
    X_test, mask_test, time_encoding_test, etas_test = build_dataset(
        X_test_raw, s_test, e_test, d_test, t_test, eta_test, scaler_ri, scaler_latlng, scaler_len, args.max_len
    )

    np.savez(
        os.path.join(args.output_dir, "processed_data_lade.npz"),
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        mask_train=mask_train,
        mask_val=mask_val,
        mask_test=mask_test,
        time_encoding_train=time_encoding_train,
        time_encoding_val=time_encoding_val,
        time_encoding_test=time_encoding_test,
        eta_train=etas_train,
        eta_val=etas_val,
        eta_test=etas_test,
        traj_train_raw=np.array(traj_train_raw, dtype=object),
        traj_val_raw=np.array(traj_val_raw, dtype=object),
        traj_test_raw=np.array(traj_test_raw, dtype=object),
    )

    print(f"Saved processed data to {args.output_dir}.")
    print(f"Train/Val/Test sizes: {len(X_train)}, {len(X_val)}, {len(X_test)}")


if __name__ == "__main__":
    main()
