import pandas as pd
import numpy as np
from glob import glob
from scipy.signal import detrend
import argparse
from pathlib import Path
from NewCrashRepair import CrashRepair
import matplotlib.pyplot as plt
def zscale(series):
    return (series - series.mean()) / series.std()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--zscale_vectors", action="store_true", required=False)
    parser.add_argument("--detrend_vectors", action="store_true", required=False)
    return parser.parse_args()

def compute_velocity(df, target_col):
    df[f"{target_col}_vel"] = df[target_col].diff().fillna(0) * df['flip_time'].diff().fillna(0) * 1000

# def resample_data(data, target_frequency=30):
#     new_time_index = np.arange(data['flip_time'].iloc[0], data['flip_time'].iloc[-1], 1.0 / target_frequency)
#     resampled_data = pd.DataFrame({
#         'flip_time': new_time_index,
#         'stim_pos': np.interp(new_time_index, data['flip_time'], data['stim_pos']),
#         'user_pos': np.interp(new_time_index, data['flip_time'], data['user_pos'])
#     })
#     return resampled_data

def process_file(file_path, output_path, detrend_vectors, zscale_vectors):
    df = pd.read_csv(file_path)
    df.user_pos = df.user_pos * -1
    cr = CrashRepair(df)
    cr.set_target_max_position() # Set the reset value for crash repair based 
                                 # on the user's data distribution.
    repaired_df = cr.repair_tracking()
    if df.crash_count.max() > 0:
        fig = cr.plot_repair(repaired_df, segment_index=0)
        if fig is not None:
            fig.savefig(output_path / file_path.name.replace(".csv", "_repaired.png"))
            plt.close()
        else:
            print("No crash report generated")
    df = repaired_df
    df["tracking"] = df.user_pos - df.stim_pos
    df["covary"] = np.abs(df.user_pos) - np.abs(df.stim_pos)
    df["abs_tracking"] = np.abs(df.tracking)
    df["abs_covary"] = np.abs(df.covary)

    for col in ["user_pos", "stim_pos", "tracking"]:
        compute_velocity(df, col)

    if detrend_vectors:
        for col in df.columns:
            if col != "flip_time":
                df[col] = detrend(df[col])

    if zscale_vectors:
        for col in df.columns:
            if col != "flip_time":
                df[col] = zscale(df[col])

    # df = resample_data(df)
    
    # Annotate filename with tags
    filename = file_path.name.replace(".csv", "")
    if detrend_vectors:
        filename += "_detrend"
    if zscale_vectors:
        filename += "_zscale"
    filename += ".csv"
    
    df.user_pos = df.user_pos * -1
    df.to_csv(output_path / filename, index=False)

def main():
    args = parse_arguments()
    base_path = Path(args.base_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path in base_path.glob("*.csv"):
        print(file_path)
        process_file(file_path, output_path, args.detrend_vectors, args.zscale_vectors)

if __name__ == "__main__":
    main()


# Test command: python3 /Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/IRT_extraction/reproc_cpCST.py --base_path '/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/IRT_extraction/proc_IRT_data' --output_path '/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/IRT_extraction/proc_IRT_data/reproc_irt_data'
'/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/IRT_extraction/proc_IRT_data/sub-M10909540_ses-MOBI1A_task-CPT_run-1_events.csv'
'/Users/danielgarcia-barnett/Desktop/Coding/cpCST_data_analysis/IRT_extraction/proc_IRT_data/reproc_irt_data'