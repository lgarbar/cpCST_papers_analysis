import mne
import pandas as pd
import argparse
import os
import numpy as np
from tkinter import Tk, filedialog

def get_fif_path():
    Tk().withdraw()
    return filedialog.askopenfilename(title="Select FIF file", filetypes=[("FIF files", "*.fif")])

def get_csv_dest_path():
    Tk().withdraw()
    return filedialog.askdirectory(title="Select destination folder")

parser = argparse.ArgumentParser(description='Open and plot an MNE FIF file.')
parser.add_argument('fif_fpath', nargs='?', type=str, help='Path to the FIF file')
parser.add_argument('--csv_dest', type=str, help='Path to the destination folder for the CSV file')
args = parser.parse_args()

fif_fpath = args.fif_fpath if args.fif_fpath else get_fif_path()
csv_dest = args.csv_dest if args.csv_dest else get_csv_dest_path()

if csv_dest:
    if not csv_dest.endswith(os.path.sep):
        csv_dest += os.path.sep
    csv_fpath = os.path.join(csv_dest, os.path.basename(fif_fpath).replace('.fif', '.csv'))
else:
    csv_fpath = os.path.join(os.path.dirname(fif_fpath), os.path.basename(fif_fpath).replace('.fif', '.csv'))

fif_obj = mne.io.read_raw_fif(fif_fpath, preload=True)

data, times = fif_obj[:]
channel_names = fif_obj.info['ch_names']


df = pd.DataFrame(data.T, columns=channel_names)
df['annot'] = np.nan

annotations = fif_obj.annotations
for ann in annotations:
    onset = ann['onset']
    description = ann['description']
    if 'Sync' not in description:
        # converting onset time to index
        idx = int(onset * fif_obj.info['sfreq'])
        if idx < len(df):
            df.at[idx, 'annot'] = description

df.to_csv(csv_fpath, index=False)
print(f"CSV file saved to: {csv_fpath}")