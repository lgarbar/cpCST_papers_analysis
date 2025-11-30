import numpy as np
from scipy.interpolate import PchipInterpolator
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class CrashRepair:
    def __init__(self, data_df, sampling_rate=30, target_max_position=0.4, window_size=3.0):
        self.data = data_df.copy()
        self.sampling_rate = sampling_rate
        self.frame_duration = 1 / sampling_rate
        self.target_max_position = target_max_position
        self.window_frame_count = int(window_size * sampling_rate)

    def set_target_max_position(self):
        abs_stim = np.abs(self.data['stim_pos'].values)
        sorted_abs = np.sort(abs_stim)
        tgt_idx = int(len(sorted_abs) * 0.99)
        self.target_max_position = sorted_abs[tgt_idx]

    def smooth_dampen(self, values, target_max, scale_factor=1.5):
        normalized = np.abs(values) / target_max
        dampened = np.tanh(normalized / scale_factor) * target_max
        return np.sign(values) * dampened

    def compute_weighted_velocity(self, positions, times, window=5):
        if len(positions) < window:
            return 0
        velocities = np.diff(positions[-window:]) / np.diff(times[-window:])
        weights = np.linspace(0.5, 1.0, len(velocities))
        return np.average(velocities, weights=weights)

    def find_crash_segments(self):
        crash_transitions = self.data[self.data['crash_count'].diff() != 0].index.tolist()
        segments = []

        for crash_idx in crash_transitions:
            pre_window_size = min(crash_idx, self.window_frame_count)
            post_window_size = min(len(self.data) - crash_idx, self.window_frame_count)

            pre_start_idx = max(0, crash_idx - pre_window_size)
            post_end_idx = min(len(self.data), crash_idx + post_window_size)

            pre_crash = self.data.iloc[pre_start_idx:crash_idx]
            post_crash = self.data.iloc[crash_idx:post_end_idx]

            if len(pre_crash) == 0 or len(post_crash) == 0:
                continue

            gap_duration = post_crash['flip_time'].iloc[0] - pre_crash['flip_time'].iloc[-1]
            n_missing_frames = int(np.round(gap_duration * self.sampling_rate))

            segments.append({
                'crash_idx': crash_idx,
                'pre_crash': pre_crash,
                'post_crash': post_crash,
                'gap_duration': gap_duration,
                'n_missing_frames': n_missing_frames
            })

        return segments

    def detect_aberrant_data(self, data, threshold=1.5):
        diff_data = np.abs(np.diff(data))
        mean_diff = np.mean(diff_data)
        std_diff = np.std(diff_data)
        aberrant_flags = diff_data > (mean_diff + threshold * std_diff)

        aberrant_points = np.zeros(len(data), dtype=bool)

        # Ensure the aberrant_flags have the same length as aberrant_points
        if len(aberrant_flags) < len(data) - self.window_frame_count:
            # Pad aberrant_flags if necessary
            aberrant_flags = np.concatenate([aberrant_flags, [False] * (len(data) - len(aberrant_flags))])

        # Pad the aberrant_flags to match the length of aberrant_points
        aberrant_flags = np.pad(aberrant_flags, (0, len(data) - len(aberrant_flags)), 'constant', constant_values=False)

        # Now assign the aberrant_flags
        aberrant_points[self.window_frame_count:] = aberrant_flags

        # Mark the first and last window_frame_count indices as aberrant
        aberrant_points[:self.window_frame_count] = True
        aberrant_points[-self.window_frame_count:] = True

        return aberrant_points

    def compute_transition(self, pre_data, post_data):
        if len(pre_data) < 2 or len(post_data) < 2:
            return None

        pre_times = pre_data['flip_time'].values
        post_times = post_data['flip_time'].values
        pre_stim = pre_data['stim_pos'].values
        post_stim = post_data['stim_pos'].values
        pre_user = pre_data['user_pos'].values
        post_user = post_data['user_pos'].values

        times = np.concatenate([pre_times, post_times])
        stim_positions = np.concatenate([pre_stim, post_stim])
        user_positions = np.concatenate([pre_user, post_user])

        unique_times, unique_indices = np.unique(times, return_index=True)
        stim_positions = stim_positions[unique_indices]
        user_positions = user_positions[unique_indices]

        sorted_indices = np.argsort(unique_times)
        sorted_times = unique_times[sorted_indices]
        sorted_stim = stim_positions[sorted_indices]
        sorted_user = user_positions[sorted_indices]

        stim_pchip = PchipInterpolator(sorted_times, sorted_stim)
        user_pchip = PchipInterpolator(sorted_times, sorted_user)

        interp_stim = stim_pchip(sorted_times)
        interp_user = user_pchip(sorted_times)

        interp_stim = self.smooth_dampen(interp_stim, self.target_max_position)
        interp_user = self.smooth_dampen(interp_user, self.target_max_position)

        window_length = min(15, len(interp_stim) // 2 * 2 - 1)
        if window_length > 3:
            interp_stim = savgol_filter(interp_stim, window_length, 3)
            interp_user = savgol_filter(interp_user, window_length, 3)

        return pd.DataFrame({
            'stim_pos': interp_stim,
            'user_pos': interp_user,
            'flip_time': sorted_times
        })

    def resample_data(self, data, target_frequency=30):
        # Convert flip_time to numeric and coerce errors to NaN
        data['flip_time'] = pd.to_numeric(data['flip_time'], errors='coerce')

        # Check for rows where flip_time is NaN (invalid values)
        if data['flip_time'].isnull().any():
            print("Warning: Some flip_time values are invalid (non-numeric). These will be replaced with NaN.")
            # Optionally, drop rows with invalid flip_time values
            data = data.dropna(subset=['flip_time'])
            # Alternatively, you could fill NaN with a specific value:
            # data['flip_time'] = data['flip_time'].fillna(method='ffill')  # Forward fill
            # data['flip_time'] = data['flip_time'].fillna(method='bfill')  # Backward fill

        # Ensure flip_time is numeric after cleaning
        start_time = data['flip_time'].iloc[0]
        end_time = data['flip_time'].iloc[-1]

        # Ensure start_time < end_time and there's a valid range
        if start_time >= end_time:
            raise ValueError(f"Start time {start_time} is greater than or equal to end time {end_time}.")

        # Generate new time index
        time_range = end_time - start_time
        if time_range <= 0:
            raise ValueError("Invalid time range. Ensure there is a positive difference between start_time and end_time.")

        new_time_index = np.arange(start_time, end_time, 1.0 / target_frequency)

        # Proceed with interpolation
        stim_interp = np.interp(new_time_index, data['flip_time'], data['stim_pos'])
        user_interp = np.interp(new_time_index, data['flip_time'], data['user_pos'])

        # Creating the resampled dataframe
        resampled_data = pd.DataFrame({
            'flip_time': new_time_index,
            'stim_pos': stim_interp,
            'user_pos': user_interp
        })

        return resampled_data

    def repair_tracking(self):
        segments = self.find_crash_segments()
        repaired_data = self.data.copy()

        aberrant_flags = self.detect_aberrant_data(self.data['stim_pos'])
        repaired_data.loc[aberrant_flags, ['stim_pos', 'user_pos']] = np.nan
        # Fixing length mismatch by adjusting the transition_df length
        # Fixing length mismatch by adjusting the transition_df length
        for segment in segments:
            transition_df = self.compute_transition(segment['pre_crash'], segment['post_crash'])

            if transition_df is None:
                continue

            start_idx = segment['pre_crash'].index[0]
            end_idx = segment['post_crash'].index[-1]

            # Ensure the lengths are compatible
            segment_length = len(repaired_data.loc[start_idx:end_idx])
            transition_length = len(transition_df)

            if segment_length != transition_length:
                min_length = min(segment_length, transition_length)
                transition_df = transition_df.head(min_length)

            # If segment_length is greater, trim repaired_data
            elif segment_length > transition_length:
                repaired_data.loc[start_idx:end_idx] = repaired_data.loc[start_idx:end_idx].head(transition_length)
            
            # Align the transition_df's length to match repaired_data's slice
            transition_df = transition_df.reindex(repaired_data.loc[start_idx:end_idx].index)

            # Now assign the values
            repaired_data.loc[start_idx:end_idx, ['stim_pos', 'user_pos']] = transition_df[['stim_pos', 'user_pos']].values
            repaired_data.loc[start_idx:end_idx, 'flip_time'] = transition_df['flip_time'].values
            repaired_data.loc[start_idx:end_idx, 'crash_count'] = 0

        repaired_data['stim_pos'] = repaired_data['stim_pos'].interpolate()
        repaired_data['user_pos'] = repaired_data['user_pos'].interpolate()
        repaired_data = self.resample_data(repaired_data, target_frequency=30)

        return repaired_data

    def plot_repair(self, repaired_data, segment_index=0):
        segments = self.find_crash_segments()
        if segment_index >= len(segments):
            return None

        segment = segments[segment_index]
        crash_idx = segment['crash_idx']
        window_size = int(2 * self.window_frame_count)
        start_idx = max(0, crash_idx - window_size)
        end_idx = min(len(self.data), crash_idx + window_size)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(self.data['flip_time'], self.data['stim_pos'], 'r--', label='Original Stimulus')
        ax1.plot(repaired_data['flip_time'], repaired_data['stim_pos'], 'b-', label='Repaired Stimulus')
        ax1.set_ylabel('Stimulus Position')
        ax1.legend()

        ax2.plot(self.data['flip_time'], self.data['user_pos'], 'r--', label='Original User')
        ax2.plot(repaired_data['flip_time'], repaired_data['user_pos'], 'b-', label='Repaired User')
        ax2.set_ylabel('User Position')
        ax2.legend()

        plt.tight_layout()
        return fig