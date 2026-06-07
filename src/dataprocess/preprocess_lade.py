import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm


def vectorized_distance(df):
    if len(df) == 0:
        return pd.Series([], dtype=float)
    lats = df["lat"].values
    lngs = df["lng"].values
    dist = np.empty(len(df))
    dist[0] = 0
    dist[1:] = np.sqrt(np.square(lats[1:] - lats[:-1]) + np.square(lngs[1:] - lngs[:-1]))
    return pd.Series(dist, index=df.index)


def is_stationary(df, stationary_threshold=10):
    distances = vectorized_distance(df)
    return distances <= stationary_threshold


def split_trajectory(
    group_df,
    time_threshold=1200,
    distance_threshold=1000,
    stationary_threshold=10,
    pause_time_threshold=90,
):
    if len(group_df) == 0:
        return []

    time_diffs = group_df["gps_time"].diff().dt.total_seconds().fillna(0).values
    distances = vectorized_distance(group_df).values
    stationary_flags = is_stationary(group_df, stationary_threshold).values

    pause_flags = np.zeros(len(group_df), dtype=bool)
    pause_start = None
    pause_time = 0

    for i in range(1, len(group_df)):
        if stationary_flags[i]:
            if pause_start is None:
                pause_start = i - 1
                pause_time = time_diffs[i]
            else:
                pause_time += time_diffs[i]
        else:
            if pause_start is not None and pause_time >= pause_time_threshold:
                pause_flags[i] = True
            pause_start = None
            pause_time = 0

    split_points = (time_diffs > time_threshold) | (distances > distance_threshold) | pause_flags
    split_indices = np.where(split_points)[0]
    if len(split_indices) == 0:
        return [group_df]

    segments = []
    start = 0
    for idx in split_indices:
        if idx > start:
            segments.append(group_df.iloc[start:idx])
        start = idx + 1
    if start < len(group_df):
        segments.append(group_df.iloc[start:])
    return segments


def process_group(group_tuple):
    _, group_df = group_tuple
    return split_trajectory(group_df)


def batch_processing(batch, output_dir, base_idx, min_duration=300, min_distance=200):
    segment_counter = base_idx
    for segment in batch:
        if len(segment) <= 1:
            continue

        total_time = (segment["gps_time"].iloc[-1] - segment["gps_time"].iloc[0]).total_seconds()
        total_distance = vectorized_distance(segment).sum()
        if total_time < min_duration or total_distance < min_distance:
            continue

        segment.to_pickle(os.path.join(output_dir, f"trajectory_segment_{segment_counter}.pkl"))
        segment_counter += 1
    return segment_counter - base_idx


def main():
    parser = argparse.ArgumentParser(description="Split raw courier trajectories into per-trip pkl files.")
    parser.add_argument("--input-file", required=True, help="Input raw trajectory dataframe, e.g. .pkl.xz")
    parser.add_argument("--output-dir", required=True, help="Output directory for trajectory_segment_*.pkl")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    data = pd.read_pickle(args.input_file)
    print("Processing timestamps...")
    data["gps_time"] = pd.to_datetime("2022-" + data["gps_time"], format="%Y-%m-%d %H:%M:%S")
    data.sort_values(by=["postman_id", "gps_time"], inplace=True)

    print("Preparing groups...")
    groups = data.groupby("postman_id", sort=False)
    group_list = [(name, group) for name, group in groups]

    print("Starting parallel processing...")
    segment_counter = 1
    processed_count = 0
    with ProcessPoolExecutor() as executor:
        total_groups = len(group_list)
        for i in tqdm(range(0, total_groups, 100), desc="Processing groups"):
            batch_groups = group_list[i : i + 100]
            results = list(executor.map(process_group, batch_groups))
            all_segments = [seg for seg_list in results for seg in seg_list if len(seg) > 1]

            for j in range(0, len(all_segments), args.batch_size):
                batch = all_segments[j : j + args.batch_size]
                count = batch_processing(batch, args.output_dir, segment_counter)
                segment_counter += count
                processed_count += count

    print(f"Saved {processed_count} trajectory segments to {args.output_dir}.")


if __name__ == "__main__":
    main()
