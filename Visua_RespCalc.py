#%%
from os.path import basename, splitext
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import heartpy as hp
import neurokit2 as nk
from scipy import signal, integrate
from scipy.signal import butter, lfilter
import scipy.signal
import pywt
from datetime import datetime

#%%Read in of the signals to be processed

if __name__ == '__main__':
    description = "Pre-Processing data"
    filename = r"Beademing\Beademde_patienten\deep breath_Movesense_body 1.csv" #r"data\ppg_name_pos_pos.csv
    name = filename[29:len(filename)]
    name=name.replace(".csv","")
    df_ori = pd.read_csv(filename, sep=",")
    filename2 = r"Beademing\Beademde_patienten\deep breath_Movesense_neck 1.csv" #r"data\ppg_name_pos_pos.csv
    name2 = filename2[29:len(filename2)]
    name2=name2.replace(".csv","")
    df_ori2 = pd.read_csv(filename2, sep=",")
    view = True
    sample_rate = 208
    
    base = basename(filename)
    f, _ = splitext(base)
    output = f + '.png'

    merged_df = pd.merge(df_ori, df_ori2, left_index=True, right_index=True)

# %%Visualisation plots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.9, 
                    top=1.9, 
                    wspace=0.7, 
                    hspace=0.6)
#ax1.plot(df_ori['time'], df_ori["x"], label='x')
ax1.plot(merged_df['times_clock_body'],merged_df["ax_body"], label='x')
ax1.plot(merged_df['times_clock_body'],merged_df["ay_body"], label='y')
ax1.plot(merged_df['times_clock_body'],merged_df['az_body'], label='z')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel("Acceleration [m/s²] right")
ax1.set_xlabel("Time [s]")

#ax2.plot(df_ori['time'],df_ori['v_x'], label='x')
ax2.plot(merged_df['times_clock_body'],merged_df['ax_neck'], label='x')
ax2.plot(merged_df['times_clock_body'],merged_df['ay_neck'], label='y')
ax2.plot(merged_df['times_clock_body'],merged_df['az_neck'], label='z')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_ylabel("Acceleration [m/s²] left")
ax2.set_xlabel("Time [s]")

#ax1.plot(df_ori['time'], df_ori["x"], label='x')
ax3.plot(merged_df['times_clock_body'],merged_df["gx_body"], label='x')
ax3.plot(merged_df['times_clock_body'],merged_df["gy_body"], label='y')
ax3.plot(merged_df['times_clock_body'],merged_df['gz_body'], label='z')
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3.set_ylabel("Gyroscoop [°/s] right")
ax3.set_xlabel("Time [s]")

#ax2.plot(df_ori['time'],df_ori['v_x'], label='x')
ax4.plot(merged_df['times_clock_body'],merged_df['gx_neck'], label='x')
ax4.plot(merged_df['times_clock_body'],merged_df['gy_neck'], label='y')
ax4.plot(merged_df['times_clock_body'],merged_df['gz_neck'], label='z')
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax4.set_ylabel("Gyroscoop [°/s] left")
ax4.set_xlabel("Time [s]")

# %%filters
def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

#%% filtering
merged_df['ay_neck_fil'] = bandpass(merged_df['ay_neck'], [0.08, 0.5], sample_rate)
merged_df['ay_neck_fil'] = merged_df['ay_neck_fil'] - np.mean(merged_df['ay_neck_fil'])
merged_df['vy_neck_fil'] = scipy.integrate.cumtrapz(merged_df["ay_neck_fil"], merged_df.index/208, initial=0)
merged_df['vy_neck_fil'] = merged_df['vy_neck_fil'] - np.mean(merged_df['vy_neck_fil'])
merged_df['dy_neck_fil'] = scipy.integrate.cumtrapz(merged_df['vy_neck_fil'], merged_df.index/208, initial=0)
#merged_df['dy_neck_fil'] = merged_df['dy_neck_fil'] - np.mean(merged_df['dy_neck_fil'])
# %%Visualisation plots
fig, (ax1) = plt.subplots(1)
plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.9, 
                    top=1.9, 
                    wspace=0.7, 
                    hspace=0.6)
#ax1.plot(df_ori['time'], df_ori["x"], label='x')
#ax1.plot(merged_df.index/208,merged_df["ay_neck"], label='acc')
#ax1.plot(merged_df.index/208,merged_df["ay_neck_fil"], label='acc_filtered')
ax1.plot(merged_df.index/208, merged_df['vy_neck_fil'], label='veloc_filtered')
#ax1.plot(merged_df.index/208, merged_df['dy_neck_fil'], label='distance_filtered')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel("Acceleration [m/s²] right")
ax1.set_xlabel("Time [s]")

# %%
