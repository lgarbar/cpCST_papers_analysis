import mne
import argparse
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
# plt.switch_backend('TkAgg') # Uncomment if MNE crashes while trying to plot.
# May need to change the visualization backend tools

def get_fif_path():
    Tk().withdraw()
    return filedialog.askopenfilename(title="Select FIF file", filetypes=[("FIF files", "*.fif")])

parser = argparse.ArgumentParser(description='Open and plot an MNE FIF file.')
parser.add_argument('fif_fpath', nargs='?', type=str, help='Path to the FIF file')
args = parser.parse_args()

fif_fpath = args.fif_fpath if args.fif_fpath else get_fif_path()

fif_obj = mne.io.read_raw_fif(fif_fpath, preload=True)
print('Plotting FIF file')
fif_obj.plot(block=True)