import os
nthreads = "8"

# Set the number of threads for various libraries
os.environ["OMP_NUM_THREADS"] = nthreads
os.environ["OPENBLAS_NUM_THREADS"] = nthreads
os.environ["MKL_NUM_THREADS"] = nthreads
os.environ["VECLIB_MAXIMUM_THREADS"] = nthreads
os.environ["NUMEXPR_NUM_THREADS"] = nthreads
os.environ["NUMBA_NUM_THREADS"] = nthreads
os.environ["NUMBA_DEFAULT_NUM_THREADS"] = nthreads
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["GOTO_NUM_THREADS"] = nthreads
os.environ["OMP_THREAD_LIMIT"] = nthreads
os.environ["BLIS_NUM_THREADS"] = nthreads
os.environ["PTHREAD_POOL_SIZE"] = nthreads

import argparse
import pickle
import pandas as pd
import mne
import numpy as np
import re
from datetime import datetime, timezone
import matplotlib.pyplot as plt

# Utility functions
def is_even(s):
    match = re.search(r'\d+', s)
    if match:
        number = int(match.group())
        return number % 2 == 0
    
def remove_numerals(s):
    return s.translate(str.maketrans('', '', '0123456789'))

anatomy_ref_dict = {2: {'Fp': 'frontopolar', 'AF': 'frontal', 'FC':'fronto-central', 'TP':'temporal-parietal', 'CP': 'centro-parietal', 'PO': 'parieto-occipital', 'FT': 'fronto-temporal'}, 
                    1: {'F': 'frontal', 'P': 'parietal', 'I': 'occipital', 'O': 'occipital', 'C': 'central'}}

def anatomy_and_laterality(channel_names, dict_template):
    channel_name_template = dict_template['labels']['channels_info']['channel_name']
    anatomy_name_template = dict_template['labels']['channels_info']['anatomy']
    laterality_name_template = dict_template['labels']['channels_info']['laterality']

    anatomy = []
    for ch in channel_names:
        if ch in channel_name_template:
            anatomy.append(anatomy_name_template[channel_name_template.index(ch)])
        elif 'z' in ch:
            anatomy.append(anatomy_ref_dict[len(ch_val)][ch_val])
        else:
            if 'z' in ch:
                ch = ch.replace('z', '')
            ch_val = remove_numerals(ch)
            anatomy.append(anatomy_ref_dict[len(ch_val)][ch_val])

    laterality = []
    for ch in channel_names:
        if ch in channel_name_template:
            laterality.append(laterality_name_template[channel_name_template.index(ch)])
        else:
            if 'z' in ch:
                laterality.append('midline')
            elif is_even(ch):
                laterality.append('right')
            else:
                laterality.append('left')

    return anatomy, laterality

def extract_envelopes(eeg_data: mne.io.Raw, band: str):
    bands_dict = {
        'delta': [1,4],
        'theta': [4,8],
        'alpha': [8,13],
        'beta': [13,30],
        'gamma': [30, 40]
    }
    return eeg_data.copy().filter(bands_dict[band][0],
                                  bands_dict[band][1]
                                  ).apply_hilbert(envelope = True)

class BlinkRemover:
    """This class is a helper to remove blinks from EEG using SSP projectors.

    You should initiate the object by giving as inputs the raw data (mne.Raw
    object) and the channel names on which the blinks are the most present
    (By default Fp1 and Fp2)
    """
    def __init__(self, 
                 raw: mne.io.Raw, 
                 channels: list[str] = ['Fp1', 'Fp2']):
        self.raw = raw
        self.channels = channels
    
    def _find_blinks(self: 'BlinkRemover') -> 'BlinkRemover':
        self.eog_evoked = mne.preprocessing.create_eog_epochs(self.raw, ch_name = self.channels).average()
        self.eog_evoked.apply_baseline((None, None))
        return self
    
    def plot_removal_results(self: 'BlinkRemover', 
                             saving_filename: str | os.PathLike | None = None
                             ) -> plt.figure:
        """Plot how well the blinks were removed.
        
        In a REPL when testing the BlinkRemover object it's always good to have
        a good view on how well the blinks were removed.

        Args:
            saving_filename (, optional): _description_. Defaults to None.
        """
        figure = mne.viz.plot_projs_joint(self.eog_projs, self.eog_evoked)
        figure.suptitle("EOG projectors")
        if saving_filename:
            figure.savefig(saving_filename)
        
        return figure
    
    def plot_blinks_found(self: 'BlinkRemover', 
                          saving_filename: str | os.PathLike | None = None
                          ) -> plt.figure:
        """Plot the result of the automated blink detection.

        Args:
            saving_filename (str | os.PathLike | None, optional): _description_. Defaults to None.

        Returns:
            plt.figure: The figure generated.
        """
        self._find_blinks()
        figure = self.eog_evoked.plot_joint(times = 0)
        if saving_filename:
            figure.savefig(saving_filename)
        return figure
    
    def remove_blinks(self: 'BlinkRemover') -> mne.io.Raw:
        """Remove the EOG artifacts from the raw data.

        Args:
            raw (mne.io.Raw): The raw data from which the EOG artifacts will be removed.

        Returns:
            mne.io.Raw: The raw data without the EOG artifacts.
        """
        self.eog_projs, _ = mne.preprocessing.compute_proj_eog(
            self.raw, 
            n_eeg=1,
            reject=None,
            no_proj=True,
            ch_name = self.channels
        )
        self.blink_removed_raw = self.raw.copy()
        self.blink_removed_raw.add_proj(self.eog_projs).apply_proj()
        return self          


def main(eeg_fpath, events_fpath, dict_fpath_template, dict_outpath):
    try:
        # Import the participant's data
        eeg_obj = mne.io.read_raw_fif(eeg_fpath, preload=True)
        blink_remover = BlinkRemover(eeg_obj)
        blink_remover.remove_blinks()
        eeg_obj = blink_remover.blink_removed_raw

        events_obj = pd.read_csv(events_fpath)

        # Grab a template pkl file to match up the eeg electrode information
        with open(dict_fpath_template, 'rb') as file:
            dict_template = pickle.load(file)

        data_dict = {}

        # Adding the eeg data to a dictionary
        bands = ['theta', 'delta', 'alpha', 'beta', 'gamma']
        eeg_features = list()
        eeg_time = eeg_obj.times

        for band in bands:
            envelope = extract_envelopes(eeg_obj, band)
            # Get the data and add a new axis to make it 3D
            eeg_band_data = envelope.get_data()
            eeg_band_data = np.expand_dims(eeg_band_data, axis=2)  # Add a new axis for the band
            eeg_features.append(eeg_band_data)

        # Now concatenate along axis 2 (which represents different bands)
        eeg_features = np.concatenate(eeg_features, axis=2)
        channel_names = (eeg_obj.info['ch_names'])
        anatomy, laterality = anatomy_and_laterality(channel_names, dict_template)
        eeg_indices = list(range(len(channel_names)))

        meas_date = eeg_obj.info['meas_date']
        ptp_num = eeg_fpath.split('/')[-1].split('_')[0].split('-')[-1]
        data_dict['eeg_data'] = {'time_info': {'time': eeg_time, 'meas_date': meas_date}, 
                                 'labels': {'channels_info': {'index': eeg_indices, 
                                                              'channels_name': channel_names, 
                                                              'anatomy': anatomy, 
                                                              'laterality': laterality},
                                            'frequency_bands': bands
                                                              }, 
                                 'features': eeg_features,
                                 'features_info': f'EEG data for participant {ptp_num}'
                                 }

        # Adding the events data to a dictionary
        events_time = [round((datetime.fromtimestamp(ts, timezone.utc) - meas_date).total_seconds(), 3) for ts in events_obj.timestamps]
        events_indices = list(events_obj.index)
        events_labels = events_obj.StimMarkers_alpha.values
        events_features = [1 if 'crash' in event.lower() else 0 for event in events_labels]
        data_dict['events_data'] = {'time': events_time, 
                                    'labels': {'index': events_indices, 
                                               'event_labels': events_labels}, 
                                    'features': events_features, 
                                    'features_info': f'Events data for participant {ptp_num}. Binary classification: 1 = Crash, 0 = Other event marker'}

        with open(dict_outpath, 'wb') as file:
            pickle.dump(data_dict, file)  

        return True     

    except Exception as e:
        print(f"Error: {e}")
        return False          

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process EEG and events data.')
    parser.add_argument('eeg_fpath', type=str, help='Path to the EEG data file (FIF format).')
    parser.add_argument('events_fpath', type=str, help='Path to the events data file (CSV format).')
    parser.add_argument('dict_fpath_template', type=str, help='Path to the template pickle file.')
    parser.add_argument('dict_outpath', type=str, help='Path to the output pickle file.')

    args = parser.parse_args()

    main(args.eeg_fpath, args.events_fpath, args.dict_fpath_template, args.dict_outpath)