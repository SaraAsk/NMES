#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:37:03 2021

@author: sara
"""

from scipy import stats
from scipy.signal import find_peaks


def XDF_correct_time_stamp_reject_pulses(f):
    
    
    marker = pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
    brainvision = pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
    edcix = [i for i,v in enumerate(brainvision['info']['desc'][0]['channels'][0]['channel']) if v['label'][0] == 'EDC_R'][0]
    edcdat = brainvision['time_series'][:,edcix]
    out = {'pulse_BV':[], 'drop_idx_list': []}
    
    # pulses creates a list of the indices of the marker timestamps for the stimulation condition trials only
    # i.e., excluding the vigilance task trials
    pulses = [i for i,m in enumerate(marker['time_series']) if "\"stim_type\": \"TMS\"" in m[0]]

    # pulseinfo contains a list of the stim.condition time stamps and descriptions
    # each item in the list contains a list with the size 2: pulseinfo[i][0] is the timestamp corresponding with the index i from pulses,
    # pulseinfo[i][1] contains the corresponding stimulus description (i.e., stim phase and freq, etc.)
    pulseinfo = [[np.searchsorted(brainvision['time_stamps'], marker['time_stamps'][p]), marker['time_series'][p]] for p in pulses]
    n=0
    
    for i,p in enumerate(pulseinfo):
        pulse_idx = pulses[pulseinfo.index(p)]
        sample = p[0]

        # For the NMES study, we use the ECD_R data to identify the artifact
        # and we use a time window around the onset of the original reizmarker_timestamp: [sample-1500:sample+1500]
        onset = sample-1500
        offset = sample+1500
        edcep = edcdat[onset:offset]
        dmy= np.abs(stats.zscore(edcep))
        tartifact = np.argmax(dmy)
        
        # edcep contains 3000 timepoints or samples (-1500 to +1500 samples around the original rm_marker)
        # so, if tartifact is < 1500, the new marker is in the period before the original marker
        # if tartifact is >1500, the new marker is in the period after the original marker      
        corrected_timestamp = sample - 1500 + tartifact

        #print('the original marker ts was: ' + str(sample)+' and the corrected ts is: '+str(corrected_timestamp))
        
        # the section below is to check for trials where no clear stimulation artifact is present
        # a list of indices is created and saved in out['drop_idx_list'], to be used to reject 
        # these epochs when the preprocessing in MNE is started
        if max(dmy) < 3:
            n+=1
            out['drop_idx_list'].append(pulse_idx)
        out['pulse_BV'].append(corrected_timestamp)
    _, _, pulses_ind_drop = np.intersect1d(out['drop_idx_list'], pulses, return_indices=True)

    pulses_ind_drop_filename = 'pulses_ind_drop_'+ str(f.parts[-3])+'_'+str(f.parts[-1][-8:-4])+'.p'
    with open(str(save_folder_pickle) +pulses_ind_drop_filename, 'wb') as fp:
        pickle.dump(pulses_ind_drop, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    """        
    Next, replace the original timestamps in the marker stream with the new ones.   
    - the original markers are stored in marker['time_stamps']
    - the new time stamp values are based on the brainvision['time_stamps'] values that 
    correspond with the brainvision['time_stamps'] index as stored in out['pulse_BV']
        E.g., 
        corrected_timestamp = 50961
        In [9]: brainvision['time_stamps'][50961]
        Out[9]: 374680.57453827135
        
    IMPORTANT:the values in corrected_timestamp (and pulse info) refer to the index of the timestamp, not
    the actual time value, of a timestamp in brainvision
    """
    
    marker_corrected = marker
    
    for i in range(len(pulses)):
        # for the stim.condition time stamps (corresponding to the indices stored in pulses)
        # replace original reizmarker (rm) timestamp value with the corrected timestamp value based on the EDC artifact (corrected_timestamp)
        rm_timestamp_idx = pulses[i]
        brainvision_idx = out['pulse_BV'][i]
        rm_timestamp_new_value = brainvision['time_stamps'][brainvision_idx] 
                
        print('old value: '+str(marker['time_stamps'][pulses[i]]))
        # replace original stimulus onset time stamp with the new timestamp value
        marker_corrected['time_stamps'][rm_timestamp_idx] = rm_timestamp_new_value
        print('new value: '+str(marker['time_stamps'][pulses[i]]))

        

    #### convert brainvision and corrected marker stream into a fif file that can be read by MNE ###    

    #marker_corrected = marker    #pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
    data = brainvision   #pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
    marker_corrected['time_stamps'] -= data['time_stamps'][0] #remove clock offset
    
    channel_names = [c['label'][0] for c in data['info']['desc'][0]['channels'][0]['channel'] ]
    sfreq = int(data['info']['nominal_srate'][0])
    types = ['eeg']*64
    types.extend(['emg']*(len(channel_names)-64)) #64 EEG chans, rest is EMG/EKG
    info = mne.create_info(ch_names = channel_names, sfreq = sfreq, ch_types = types)
    raw = mne.io.RawArray(data = data['time_series'].T, info = info)
    
    if len(marker_corrected['time_stamps']) > 1:
        descs = [msg[0] for msg in marker_corrected['time_series']]
        ts = marker_corrected['time_stamps']
        
        sel = [i for i,v in enumerate(descs) if "TMS" in v]
        descs = [descs[i] for i in sel]
        
        ts = [ts[i] for i in sel]
        
        shortdescs = [json.loads(msg)['freq'] + 'Hz_' + json.loads(msg)['phase'] for msg in descs]

        anno = mne.Annotations(onset = ts, duration = 0, description = shortdescs)
        raw = raw.set_annotations(anno)
        
    ts_new = np.delete(ts, pulses_ind_drop)
    shortdescs_new = np.delete(shortdescs, pulses_ind_drop)
    anno = mne.Annotations(onset = ts_new, duration = 0, description = shortdescs_new)
    raw = raw.set_annotations(anno)      
    #print(len(ts), len(ts_new))
    #print(str(f.parts[-3]))
    #cols = ['','', 'Pulses', 'Pulses Corrected']
    #lst.append(['', '',  len(ts) , len(ts_new)])
    #df1 = pd.DataFrame(lst, columns=cols)     
   # df1.to_csv ('/home/sara/NMES/analyzed data/' +'Subject_ pulse.csv', index = None, header=True)    
    #      

    return raw    



from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
import Select_Epochs as SE
import numpy as np
import scipy.signal

import pickle
from pathlib import Path
import pandas as pd
import mne
import numpy as np
import pyxdf

import json
import regression_eye_artifacts as REA

from autoreject import AutoReject
from mne.channels import make_standard_montage
import scipy.signal
import matplotlib.pyplot as plt
save_folder_pickle = "/home/sara/NMES/analyzed data/"
# from raw, unprocessed files, select the EMG channels only and make epochs. Then plot to see how well the markers match the stimulation 
# artifact

#%% Load raw data and select the EMG channels (no further preproc steps performed)
raw_data_dir = "/home/sara/NMES/NMES_Experimnet/" #replace by your own top-level directory
raw_files = Path(raw_data_dir).glob('**/*.xdf*')


evokeds_all = []
subjects_names = []
 

for f in raw_files:
    
    raw = XDF_correct_time_stamp_reject_pulses(f)
    # load raw fif file
    print('loading '+str(f.parts[5]))

    # step 1: select EMG channels    
    print('Number of channels in raw_eeg:')
    print(len(raw.ch_names), end=' → keep eight emg → ')
    raw_emg = raw.pick_channels(['EDC_R'])
    print(len(raw_emg.ch_names))
    # step x: apply filters
    sFreq = raw_emg.info['sfreq']
    raw_emg = raw_emg.notch_filter((50,100,150), picks = 'EDC_R')    # notch filter, line noise
    raw_emg = raw_emg.filter(l_freq=20, h_freq=None,  picks = 'EDC_R')    # high-pass filter at 0.5 Hz (using default filter settings = fir)  
    raw_emg = raw_emg.filter(l_freq=None, h_freq=250,  picks = 'EDC_R')    # low-pass filter at 30 Hz.
    
    # Step 2: create events from the annotations present in the raw file
    (events_from_annot, event_dict) = mne.events_from_annotations(raw_emg)
    
    # Step 3: create epochs based on the events, from -1 to 1s
    emg_epochs = mne.Epochs(raw_emg, events_from_annot, event_id=event_dict,
                        tmin= -1, tmax= 1, reject=None, preload=True,
                        baseline=None, detrend=None, event_repeated='merge')
    
    # putting ERp of EDC_R channel of all the subjects in a list
    evokeds_all.append(emg_epochs.average(picks = 'EDC_R'))
    subjects_names.append(f.parts[5])
    
    
    
# Plotting this channel for all the subjects to know how long the stimulation takes for all the subjects 
# so we can cut that part when we want to compare before and after stimulation  epochs   

# Here it starts from zero because one of the files recorded for Hasa(here number zero) was stimulated twice! at least from what i see in EDC_R   
for i in np.arange(1,len(evokeds_all)):
    #print(subjects_names[i])
    plt.plot(evokeds_all[i].times, np.transpose(evokeds_all[i]._data))
plt.axvline(x =  0.2, linestyle = 'dashed', color = 'black')
plt.axvline(x = -0.2, linestyle = 'dashed', color = 'black')

# So from what I can see from the above plot by getting rid of -0.2 to 0.2 before and after each stimulation, the stimulation arftifact will be removed. 
 
    
 