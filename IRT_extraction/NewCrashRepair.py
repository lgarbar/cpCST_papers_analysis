import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.ndimage import gaussian_filter1d

class CrashRepair:
    def __init__(self, data_df, sampling_rate=30, target_max_position=0.4, window_size=3.0):
        self.data = data_df.copy()
        self.sampling_rate = sampling_rate
        self.frame_duration = 1 / sampling_rate
        self.target_max_position = target_max_position
        self.window_frame_count = int(window_size * sampling_rate)

    def find_crash_segments(self):
        crash_transitions = self.data[self.data['crash_count'].diff() != 0].index.tolist()
        segments = []
        for crash_idx in crash_transitions:
            pre_window = min(self.window_frame_count, crash_idx)
            post_window = min(self.window_frame_count, len(self.data) - crash_idx)
            if pre_window == 0 or post_window == 0:
                continue
            pre_crash = self.data.iloc[crash_idx - pre_window:crash_idx]
            post_crash = self.data.iloc[crash_idx:crash_idx + post_window]
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

    def compute_transition(self, pre_data, post_data):
        if len(pre_data) < 2 or len(post_data) < 2:
            return None
        pre_times = pre_data['flip_time'].values
        post_times = post_data['flip_time'].values
        times = np.concatenate([pre_times, post_times])
        stim_spline = CubicSpline(times, np.concatenate([pre_data['stim_pos'].values, post_data['stim_pos'].values]), bc_type='natural')
        user_spline = CubicSpline(times, np.concatenate([pre_data['user_pos'].values, post_data['user_pos'].values]), bc_type='natural')
        interp_stim = stim_spline(times)
        interp_user = user_spline(times)
        interp_stim = gaussian_filter1d(interp_stim, sigma=2)
        interp_user = gaussian_filter1d(interp_user, sigma=2)
        return pd.DataFrame({'stim_pos': interp_stim, 'user_pos': interp_user, 'flip_time': times})

    def repair_tracking(self):
        segments = self.find_crash_segments()
        repaired_data = self.data.copy()
        for segment in segments:
            transition_df = self.compute_transition(segment['pre_crash'], segment['post_crash'])
            if transition_df is None:
                continue
            start_idx = segment['crash_idx'] - len(segment['pre_crash'])
            end_idx = segment['crash_idx'] + len(segment['post_crash'])
            if len(transition_df) != end_idx - start_idx:
                continue
            idx_range = pd.RangeIndex(start=start_idx, stop=end_idx)
            repaired_data.loc[idx_range, 'stim_pos'] = transition_df['stim_pos'].values
            repaired_data.loc[idx_range, 'user_pos'] = transition_df['user_pos'].values
            repaired_data.loc[idx_range, 'flip_time'] = transition_df['flip_time'].values
            repaired_data.loc[idx_range, 'did_crash'] = False
            last_crash_count = repaired_data.loc[start_idx-1, 'crash_count'] if start_idx > 0 else 0
            repaired_data.loc[idx_range, 'crash_count'] = last_crash_count
        return repaired_data