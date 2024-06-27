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
from scipy.signal import find_peaks, peak_prominences, welch
import pywt
from datetime import datetime

#%%Read in of the signals to be processed

if __name__ == '__main__':
    description = "Pre-Processing data"
    filename = r"ademen_zittend_guylian.csv" #r"data\ppg_name_pos_pos.csv
    name = filename[16:len(filename)]
    name=name.replace(".csv","")
    df_ori = pd.read_csv(filename, sep=",")
    view = True
 
    
    base = basename(filename)
    f, _ = splitext(base)
    output = f + '.png'


# %%
df = df_ori
df['time']=df.index/208


df['x'] =  df['x'] - np.mean(df['x'])
df['y'] =  df['y'] - np.mean(df['y'])
df['z'] =  df['z'] - np.mean(df['z'])
df['norm'] = np.linalg.norm(df[['y','z']].values,axis=1) - np.mean(np.linalg.norm(df[['x','y','z']].values,axis=1))

df['F_x']=0.01*df['x']
df['F_y']=0.01*df['y']
df['F_z']=0.01*df['z']
df['F_norm'] = np.linalg.norm(df[['F_y','F_z']],axis=1)

df['v_x'] = integrate.cumtrapz(df['x'], x = df['time'], initial=0.0)
df['v_y'] = integrate.cumtrapz(df['y'], x = df['time'], initial=0.0)
df['v_z'] = integrate.cumtrapz(df['z'], x = df['time'], initial=0.0)
df['v_norm'] = np.linalg.norm(df[['v_y','v_z']],axis=1)#-np.mean(integrate.cumtrapz(norm, x = df['time'], initial=0.0))

df['x_x'] = integrate.cumtrapz(df['v_x'], x = df['time'], initial=0.0)
df['x_y'] = integrate.cumtrapz(df['v_y'], x = df['time'], initial=0.0)
df['x_z'] = integrate.cumtrapz(df['v_z'], x = df['time'], initial=0.0)
df['x_norm'] = np.linalg.norm(df[['x_y','x_z']],axis=1)

df['E_x']=0.5*0.01*df['v_x']**2
df['E_y']=0.5*0.01*df['v_y']**2
df['E_z']=0.5*0.01*df['v_z']**2
df['E_norm'] = np.linalg.norm(df[['E_y','E_z']],axis=1)

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7)
plt.subplots_adjust(left=0.1,
                    bottom=0.2, 
                    right=0.9, 
                    top=1.9, 
                    wspace=0.7, 
                    hspace=0.6)
#ax1.plot(df['time'], df["x"], label='x')
ax1.plot(df['time'],df["y"], label='y')
ax1.plot(df['time'],df["z"], label='z')
ax1.plot(df['time'],df['norm'], label='norm')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylabel("Acceleration [m/s²]")
ax1.set_xlabel("Time [s]")

#ax2.plot(df['time'],df['v_x'], label='x')
ax2.plot(df['time'],df['v_y'], label='y')
ax2.plot(df['time'],df['v_z'], label='z')
ax2.plot(df['time'],df['v_norm'], label='norm')
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_ylabel("Speed [m/s]")
ax2.set_xlabel("Time [s]")

#ax3.plot(df['time'],df['x_x'], label='x')
ax3.plot(df['time'],df['x_y'], label='y')
ax3.plot(df['time'],df['x_z'], label='z')
ax3.plot(df['time'],df['x_norm'], label='norm')
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3.set_ylabel("Movement [m]")
ax3.set_xlabel("Time [s]")

#ax4.plot(df['time'],df['E_x'], label='x')
ax4.plot(df['time'],df['E_y'], label='y')
ax4.plot(df['time'],df['E_z'], label='z')
ax4.plot(df['time'],df['E_norm'], label='norm')
ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax4.set_ylabel("Energy [J]")
ax4.set_xlabel("Time [s]")

#ax5.plot(df['time'],df['F_x'], label='x')
ax5.plot(df['time'],df['F_y'], label='y')
ax5.plot(df['time'],df['F_z'], label='z')
ax5.plot(df['time'],df['F_norm'], label='norm')
ax5.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax5.set_ylabel("Force [N]")
ax5.set_xlabel("Time [s]")

#ax6.plot(df['time'],df['F_x'], label='x')
ax6.plot(df['time'], np.cumsum(df['F_norm']), label='cumulative Force')
ax6.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax6.set_ylabel("Force [N]")
ax6.set_xlabel("Time [s]")


ax7.plot(df['time'], np.cumsum(df['E_norm']), label='cumulative Energy')
ax7.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax7.set_ylabel("Energy [J]")
ax7.set_xlabel("Time [s]")
fig.show()


#%%
