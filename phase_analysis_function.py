#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 10:09:36 2022

@author: sara
"""


import mne
import math
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import  zscore
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

import lmfit
import itertools  
import mne.stats
from tqdm import tqdm
from pathlib import Path
from scipy.stats import sem # standard error of the mean.
from multiprocessing import Pool
from scipy.interpolate import interp1d
import phase_analysis_function as ph_analysis


lst = []
save_folder_pickle = "/home/sara/NMES/analyzed data/"

def extract_phase(filtered_data, fs, freq_band, cycles, angletype = 'degree'):
    ''' Extract the phase in degree (default) of a specific frequency band (alpha/mu or beta) with FFT
    Phase is extracted dynamically, dependending on frequencies ->
    X-cycles of each frequencies before TMS pulse are taken to get phase
    
    Args:
        filtered_data(complex):
            filtered data in epoch form 
            complex, because it's based on FFT filtered data
            epochs * times * channels   
        fs(int):
            sampling frequency   
        freq_band(string):
            name of frequency band to extract
            either 'alpha'or 'mu', or 'beta'
        cycles(int):
            based on how many cycles of frequencies the phase should be extracted      
        angletype(str):
            type of angle measurement, either 'radiant' or 'degree'
            default is to 'degree'
    
    Returns:
        float:
            eeg_epoch_filt_phase: phase of filtered epoch at sepecific frequency band
        list:
            freq_band: list of frequency-values in frequency band
                
    call: extract_phase(eeg_epoch_filtered, fs, 'beta', 3)
    '''
    
        
        
    if freq_band == '4Hz step':
        freq_band =  list(range(4, 41, 4))
    elif freq_band == '1Hz step':
        freq_band = list(range(4, 41))
    else:
        raise ValueError("Frequency band must be defined as 'alpha', 'mu' or 'beta'")  
        
    
    
    
    
    num_epoch = filtered_data.shape[0]
    eeg_epoch_filt_phase = np.nan*np.zeros((len(freq_band),num_epoch))

    for idx_epoch in range(0,num_epoch):
        for idx, value in enumerate(freq_band):
            #print(idx, value)
            respective_tw = math.ceil((1/value)*cycles*fs) + 5  # 3 cycles of frequency and 5 extra sample points
            #math.ceil: rounds a number up to the next largest integer
            cycle_tp_start = len(filtered_data[2]) - respective_tw 
            signal = filtered_data[idx_epoch, cycle_tp_start:]
            
            # put a 500ms window ?
            signal = filtered_data[idx_epoch, 0:500]
            
            
            N = len(signal) 
            fx = np.fft.rfft((signal), N)  # Extract FFT from real data. discretes Fourier Transform for real input
            T = 1.0/ fs # Timesteps
            freq = np.fft.fftfreq(N, d=T)  # Return the Discrete Fourier Transform sample frequencies.
            #idx_freq, = np.where(np.isclose(freq, value, atol=cycles+(T*N)))
            # not as straight forward
            
            # Mara's solution for indexing
            # more robust, a little bit slower
            diff = [np.abs(f-value) for f in freq]
            idx_freq = diff.index(np.min(diff))
            
            # or: is it possible to index based on cycles I look at?
            # phase = np.rad2deg(np.angle(fx)[cycles])      

# Not sure if i use the correct index here:
        # Extract phase 
        # degree -> *(np.pi/180) -> radiant
        # rad -> *(180/np.pi) -> degree
            if angletype == 'radiant':
                phase = np.angle(fx)[idx_freq] 
                eeg_epoch_filt_phase[idx,idx_epoch] = phase
                print(idx)
            
            elif angletype == 'degree':
               phase = np.rad2deg(np.angle(fx)[idx_freq])
               eeg_epoch_filt_phase[idx,idx_epoch] = phase
               
            else:
                raise ValueError("Specify angle type with either 'radiant' or 'degree'")    
          
    return eeg_epoch_filt_phase, freq_band


def assign_bin_class(epoched_phases, bin_num):
    ''' Assign each phase to the corresponding bin of phases on the unit circle
    
    Args:
        epoched_phases(float):
            phase of filtered epoch at specific frequency band
            frequencies * epochs ( * channels not yet) 
        bin_num(int):
            number of bins on unit circle that epoch phases will be assigned to
    Returns:
        float:
           bin_class: array with numbers from 0 to bin_num, corresponding to
           epoched_phases array
               frequencies * epochs
                
    call: assign_bin_class(eeg_epoch_filt_phase, bin_num = 16)
    '''    
    
# for phase as degree
    bin_anticlockwise = np.linspace(0,360,int(bin_num+1))  # cover half of the circle -> with half of bin_num
    bin_clockwise = np.linspace(-360,0,int(bin_num+1)) 


    bin_class = np.nan*np.zeros(epoched_phases.shape)

    for [row,col], phases in np.ndenumerate(epoched_phases):  
    # numbers correspond to the anti-clockwise unit circle eg. bin = 1 -> equals 22.5 deg phase for 16 bins
        if phases > 0:
                idx, = np.where(np.isclose(epoched_phases[row,col], bin_anticlockwise[:], atol=360/(bin_num*2)))
                # Returns a boolean array where two arrays are element-wise equal within a tolerance.
    # atol -> absolute tolerance level -> bin margins defined by 360° devided by twice the bin_num      
    # problem: rarely exactly between 2 bins -> insert nan
                if len(idx) > 1:
                    idx = np.nan
                bin_class[row,col] = idx
    
        elif phases < 0:
                idx, = np.where(np.isclose(epoched_phases[row,col], bin_clockwise[:], atol=360/(bin_num*2)))  
                if len(idx) > 1:
                    idx = np.nan      
                bin_class[row,col] = idx
                
                
                
                
    # bin_anticlockwise = np.linspace(0,180,int(bin_num/2+1))  # cover half of the circle -> with half of bin_num
    # bin_clockwise = np.linspace(-180,0,int(bin_num/2+1)) 
    #    # bin_clockwise = np.flip(np.linspace(-180,0,int(bin_num/2+1)))


    # bin_class = np.nan*np.zeros(epoched_phases.shape)

    # for [row,col], phases in np.ndenumerate(epoched_phases):  
    # # numbers correspond to the anti-clockwise unit circle eg. bin = 1 -> equals 22.5 deg phase for 16 bins
    #     if phases > 0:
    #             idx, = np.where(np.isclose(epoched_phases[row,col], bin_anticlockwise[:], atol=180/(bin_num)))
    # # atol -> absolute tolerance level -> bin margins defined by 360° devided by twice the bin_num      
    # # problem: rarely exactly between 2 bins -> insert nan
    #             if len(idx) > 1:
    #                 idx = np.nan
    #             bin_class[row,col] = idx
    
    #     elif phases < 0:
    #             idx, = np.where(np.isclose(epoched_phases[row,col], bin_clockwise[:], atol=180/(bin_num)))  
    #             if len(idx) > 1:
    #                 idx = np.nan      
    #             bin_class[row,col] = idx*2
    # PROBLEM -> 0 and -180 get the same bin class of 0
            
            
    return bin_class

def get_phase_bin_mean(epoched_phases, bin_class, bin_num, eeg_anticlockwise_phase = True):
    ''' take the values for each bin and average them to get the mean of each phase-bin
    
    Args:
        epoched_phases(float):
            phase of filtered epoch at sepecific frequency band
            frequencies * epochs ( * channels not yet)
        bin_class (float):
            array with values of bin numbers, corresponding to the phases in epoched_phases
            frequencies * epochs   
        bin_num(int):
            number of bins on unit circle that epoch phases will be assigned to     
        eeg_anticlockwise_phase(bool):
            if True: returns also an array with anti-clockwise(only positive) 
            values of phases; default is False  
        
    Returns:
        list:
            phase_bin_means: list of means of each trial in corresponding bin
    Optional:
        float:
            eeg_phase: only positive values for phase 
            (for an anticlockwise unit circel)
    
    call: get_phase_bin_mean(eeg_epoch_filt_phase, bin_class, bin_num = 16)
  
    '''
    # get mean of bins with vector in complex space
    # x = np.randmom.random(10)  * 2* np.pi
    # np.angle(np.exp(1j * x).mean())

    # make surethat same bins (esp. 0/360°) are projected on same bin)
    eeg_phase  =  epoched_phases.copy()
    for [r,c], value in np.ndenumerate(eeg_phase):
        if value < 0 :
            eeg_phase[r,c] = eeg_phase[r,c] + 360

     
    phase_bin_means = list(range(0,bin_num)) 

    # get radiants of values
    phase_rad = np.deg2rad(eeg_phase)
    
    # take mean for every phase bin
    # check where which values correspond to the bin_class -> take mean of all those values
    # change phase into complex number values -> np.exp(1j*phase)
    # take the mean of the vectors in complex space
    for value in list(range(0,bin_num)): 
        phase_bin_means[value] = np.angle(np.exp(1j * phase_rad[np.where(bin_class[:,:] == value)]).mean())
   
    # bin 0 and last bin have to be combined, both sit at same side 0°/360°
        if value == 0:
             bin_0 = phase_rad[np.where(bin_class[:,:] == value)]
             bin_0_360 = np.append(bin_0, phase_rad[np.where(bin_class[:,:] == bin_num)]) 
             phase_bin_means[value] = np.angle(np.exp(1j * bin_0_360).mean())
    
    # get phase value back in degrees
    phase_bin_means = np.rad2deg(phase_bin_means)
   
    return phase_bin_means










def get_phase_bin_mean_each_freq(target_freq, epoched_phases, bin_class, bin_num, eeg_anticlockwise_phase = True):
    ''' take the values for each bin and average them to get the mean of each phase-bin in each freq
    
    Args:
        epoched_phases(float):
            phase of filtered epoch at sepecific frequency band
            frequencies * epochs ( * channels not yet)
        bin_class (float):
            array with values of bin numbers, corresponding to the phases in epoched_phases
            frequencies * epochs   
        bin_num(int):
            number of bins on unit circle that epoch phases will be assigned to     
        eeg_anticlockwise_phase(bool):
            if True: returns also an array with anti-clockwise(only positive) 
            values of phases; default is False  
        
    Returns:
        list:
            phase_bin_means: list of means of each trial in corresponding bin
    Optional:
        float:
            eeg_phase: only positive values for phase 
            (for an anticlockwise unit circel)
    
    call: get_phase_bin_mean(eeg_epoch_filt_phase, bin_class, bin_num = 16)
  
    '''
    # get mean of bins with vector in complex space
    # x = np.randmom.random(10)  * 2* np.pi
    # np.angle(np.exp(1j * x).mean())

    # make surethat same bins (esp. 0/360°) are projected on same bin)
    eeg_phase  =  epoched_phases.copy()
    for [r,c], value in np.ndenumerate(eeg_phase):
        if value < 0 :
            eeg_phase[r,c] = eeg_phase[r,c] + 360

     
    phase_bin_means = np.zeros([len(target_freq), bin_num])

    # get radiants of values
    phase_rad = np.deg2rad(eeg_phase)
    
    # take mean for every phase bin
    # check where which values correspond to the bin_class -> take mean of all those values
    # change phase into complex number values -> np.exp(1j*phase)
    # take the mean of the vectors in complex space
    for i,freq in enumerate(target_freq):
        print(i,freq)
    
        for value in list(range(0,bin_num)): 
            phase_bin_means[i, value] = np.angle(np.exp(1j * phase_rad[i,:][np.where(bin_class[i,:] == value)]).mean())
       
        # bin 0 and last bin have to be combined, both sit at same side 0°/360°
            if value == 0:
                 bin_0 = phase_rad[i,:][np.where(bin_class[i,:] == value)]
                 bin_0_360 = np.append(bin_0, phase_rad[i,:][np.where(bin_class[i,:] == bin_num)]) 
                 phase_bin_means[i, value] = np.angle(np.exp(1j * bin_0_360).mean())
        
        # get phase value back in degrees
        phase_bin_means = np.rad2deg(phase_bin_means)
   
    return phase_bin_means






def plot_phase_bins(bin_means, bin_class, bin_num, scaling_proportion):
    ''' plots the mean of phase-bins and also plots the bins with the height 
    representing the number of trials within the bin (in proportion)
    
    
    Args:
        bin_means(list):
            list of means of each trial in corresponding bin
        bin_class (float):
            array with values of bin numbers, corresponding to the phases in epoched_phases
            frequencies * epochs
        bin_num(int):
            number of bins on unit circle that epoch phases will be assigned to
        scaling_proportion(int):
            an integer that scales the height of bins in order to adjust to 
            unit circle with radius = 1
            e.g. amount of trials in bin 5 = 230 -> scale by 300: 230/300 = 0.7667
            -> this bin will have a height of 0.7667 on unit circle when plotted
        
    Returns:
        plotted phases within corresponding bin
        also: height of displayed bin shows number of trials within bin (higher -> more trials)
    
    call: plot_phase_bins(bin_means, bin_class, bin_num, scaling_proportion = 300)
    '''
    theta = np.linspace(0.0, 2 * np.pi, bin_num, endpoint=False)
    rad_height = np.nan*np.zeros(bin_num+1)  # +1 because bin 0 and 16 are not combined yet
    unique, counts = np.unique(bin_class, return_counts=True) 
    # how many cases are there for each unique bin? -> dictionary of unique values of array bin_class
 
    for idx in range(unique.size):
        if unique[idx].is_integer() is True:  # Only for intger values (so, no nan due to phase in the middle of 2 bins)
               rad_height[idx] = counts[idx]/scaling_proportion  # Set height in proportion
               if unique[idx] == 0:  # For bin 0 and last bin, combine values (0/360°)
                   rad_height[0] = (counts[0]+counts[-1])/scaling_proportion
        
    # Exclude bins with nan value and the last bin (already combined with first bin)    
    nan_array = ~(np.isnan(rad_height))
    rad_height = rad_height[nan_array]
    rad_height = rad_height[:-1] 
    
    R1 = [0,1]  # Defined as UNIT circle (radius = 1)
    bin_phase_rad = np.deg2rad(bin_means)
    
    plt.figure()
    plt.polar([bin_phase_rad, bin_phase_rad], R1, lw=2, color = 'navy')
    width = (2*np.pi) / bin_num
    ax = plt.subplot(111, projection='polar')
    bars = ax.bar(theta, rad_height, width=width, bottom=0.0, color = 'lightgrey' , edgecolor = 'grey')

    return (plt)










# =============================================================================
from scipy import stats
import pyxdf


def XDF_correct_time_stamp_reject_pulses_bip(f):
    
    
    marker = pyxdf.load_xdf(f, select_streams=[{'name': 'reiz_marker_sa'}])[0][0]
    brainvision = pyxdf.load_xdf(f, select_streams=[{'name': 'BrainVision RDA'}])[0][0]
    edcix = [i for i,v in enumerate(brainvision['info']['desc'][0]['channels'][0]['channel']) if v['label'][0] == 'EDC_R'][0]
    edcdat = brainvision['time_series'][:,edcix]
    out = {'pulse_BV':[], 'drop_idx_list': []}
    # bipolar signal near to C3
    bipolar = pyxdf.load_xdf(f, select_streams=[{'name': 'Spongebob-Data'}])[0][0]
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
                
        #print('old value: '+str(marker['time_stamps'][pulses[i]]))
        # replace original stimulus onset time stamp with the new timestamp value
        marker_corrected['time_stamps'][rm_timestamp_idx] = rm_timestamp_new_value
        #print('new value: '+str(marker['time_stamps'][pulses[i]]))

        

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
    info_bi =  mne.create_info(13, sfreq=info['sfreq'])
    raw_bip = mne.io.RawArray(bipolar['time_series'].T, info = info_bi).set_annotations(anno)     
    # Index of bipolar channel
    #raw_bip = raw_bip.pick_channels(['1'])
    #print(len(ts), len(ts_new))
    #print(str(f.parts[-3]))
    cols = ['','', 'Pulses', 'Pulses Corrected']
    lst.append(['', '',  len(ts) , len(ts_new)])
    df1 = pd.DataFrame(lst, columns=cols)     
    df1.to_csv ('/home/sara/NMES/analyzed data/' +'Subject_ pulse.csv', index = None, header=True)    
    return  raw_bip, f.parts                  








def get_ERP_1st_2nd(plot):
    import matplotlib.pyplot as plt
    import numpy as np
    import mne
    import os
    evoked_all_condition_dir = "/home/sara/NMES/analyzed data/Feb/evokeds/All_Conditions_Combined/"
    evoked_all_condition_data = os.path.join(evoked_all_condition_dir,'Evokeds_all_conditions_ave.fif')
    Evoked_GrandAv= mne.read_evokeds(evoked_all_condition_data, verbose=False)
    Evoked_GrandAv = Evoked_GrandAv[0]    

    
    # This n1, n2, p2 positions needs to be modified.
    n1_timewin = [.02, .08]
    n2_timewin= [.108, .266]
    p2_timewin = [.28, .36]
    

    
    Evoked_GrandAv_mean = Evoked_GrandAv.copy().data.std(axis=0, ddof=0)
    
    idx_n1 = np.logical_and(Evoked_GrandAv.times>n1_timewin[0], Evoked_GrandAv.times<n1_timewin[1])
    idx_n2 = np.logical_and(Evoked_GrandAv.times>n2_timewin[0], Evoked_GrandAv.times<n2_timewin[1])
    idx_p2 = np.logical_and(Evoked_GrandAv.times>p2_timewin[0], Evoked_GrandAv.times<p2_timewin[1])
    
    n1 = np.max(Evoked_GrandAv_mean[idx_n1])
    tn1 = Evoked_GrandAv.times[idx_n1][Evoked_GrandAv_mean[idx_n1].argmax()]
    n2 = np.max(Evoked_GrandAv_mean[idx_n2])
    tn2 = Evoked_GrandAv.times[idx_n2][Evoked_GrandAv_mean[idx_n2].argmax()]
    #p2 = np.max(Evoked_GrandAv_mean[idx_p2])
    #tp2 = Evoked_GrandAv.times[idx_p2][Evoked_GrandAv_mean[idx_p2].argmax()]
    

    index_n1 = np.where(Evoked_GrandAv.times == tn1)
    index_n2 = np.where(Evoked_GrandAv.times == tn2)
    #index_p2 = np.where(Evoked_GrandAv.times == tp2)    
    if plot:
        fig1 = mne.viz.plot_compare_evokeds(Evoked_GrandAv)     
        fig = plt.figure(figsize=[7,5])
        ax = fig.add_subplot(111, autoscale_on=True)
        plt.plot(Evoked_GrandAv.times,Evoked_GrandAv_mean,'k-',lw=1)
        
        plt.xlabel('time (s)',fontsize = 13)
        plt.ylabel('Amplitude',fontsize = 13)
        
        #Evoked_GrandAv.plot_topomap([tn1, tn2], ch_type='eeg', time_unit='s', ncols=8, nrows='auto', vmin = -5500000, vmax = 5500000)
        #plt.figure()
        fig2, ax =  plt.subplots(ncols=2, figsize=(8, 4))
        im, cm = mne.viz.plot_topomap(Evoked_GrandAv.data[:, index_n1[0][0]], Evoked_GrandAv.info, axes=ax[0], show=False, vmin = -2, vmax= 2)
        im, cm = mne.viz.plot_topomap(Evoked_GrandAv.data[:, index_n2[0][0]], Evoked_GrandAv.info, axes=ax[1], show=False, vmin = -2, vmax= 2)
        # add titles
        ax[0].set_title('ERP 1', fontweight='bold')
        ax[1].set_title('ERP 2', fontweight='bold')
        
        
        ax_x_start = 0.92
        ax_x_width = 0.01
        ax_y_start = 0.33
        ax_y_height = 0.4
        cbar_ax = fig2.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im,  cax=cbar_ax)

        
        #all_times = np.arange(-0.2, 0.5, 0.01)
        #topo_plots = Evoked_GrandAv.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
    return([index_n1[0][0], index_n2[0][0]], Evoked_GrandAv.ch_names)



# =============================================================================



import lmfit
from tqdm import tqdm
from multiprocessing import Pool
import itertools 




def cosinus(x, amp, phi):
    return amp * np.cos(x + phi)

def unif(x, offset):
    return offset

def do_one_perm(model, params, y,x):
    resultperm = model.fit(y, params, x=x)
    return resultperm.best_values['amp']


def do_cosine_fit(erp_amplitude, phase_bin_means, freq_band, labels, perm = True):

    
    cosinefit = {}
    amplitudes_cosine = np.zeros([len(freq_band), len(labels)])
    pvalues_cosine = np.zeros([len(freq_band), len(labels)])

    
    
    x = np.radians(np.array([0, 45, 90, 135, 180, 225, 270, 315]))
    #x = phase_bin_means
    
    for i in range(len(erp_amplitude)):
        cosinefit[str(i)] = {}
        for jf, f in enumerate(freq_band):    
            print('cosine fits for frequency {}'.format(f))
            y = zscore(np.array(list(erp_amplitude[str(i)][f].values())))
            cosinefit[str(i)][str(f)] = []
            fits = []
            for phase_start in [-np.pi/2, 0, np.pi/2]:      
        
                amp_start = np.sqrt(np.mean(y**2))
                model = lmfit.Model(cosinus)
                params = model.make_params()
        
                params["amp"].set(value=amp_start, min=0, max=np.ptp(y)/2)
                params["phi"].set(value=phase_start, min=-np.pi, max=np.pi)
                data = {ph: np.mean(y[x == ph]) for ph in np.unique(x)}
                fits.append(model.fit(y, params, x=x))
                
            result = fits[np.argmin([f.aic for f in fits])]
            
            if perm:
                model = lmfit.Model(cosinus)
                params = result.params
                dataperm = []
            
                # use all possible combinations of the 8 phase bins to determine p.
                # Take out the first combination because it is the original
                all_perms = list(itertools.permutations(x))
                del all_perms[0]
            
                for iper in tqdm(range(len(all_perms))):
                    x_shuffled = all_perms[iper]
                    dataperm.append([model,params, y, x_shuffled])
            
                with Pool(4) as p:
                    surrogate = p.starmap(do_one_perm, dataperm)
            else: 
                surrogate = [np.nan]
                
            
            nullmodel = lmfit.Model(unif)
            params = nullmodel.make_params()
            params["offset"].set(value=np.mean(y), min=min(y), max=max(y))
            nullfit = nullmodel.fit(y, params, x=x)
            surrogate = np.array(surrogate)
            surrogate = surrogate[np.invert(np.isnan(surrogate))]
            
            cosinefit[str(i)][str(f)].append( { 'Model': 'cosinus', 
                                    'Frequency': f, 
                                    'Fit': result,
                                    'data': data, 
                                    'amp': result.best_values['amp'], 
                                    'surrogate': surrogate, 
                                    'p':[np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0], 
                                    'std':[np.nan if perm == False else np.std(surrogate)][0], 
                                    'nullmodel':nullfit,
                                    })
            
            amplitudes_cosine[jf, i] = result.best_values['amp']
            pvalues_cosine[jf, i] = [np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0] 
    
    
    
    return cosinefit, amplitudes_cosine, pvalues_cosine



def do_cosine_fit_ll(erp_amplitude, phase_bin_means, freq_band, labels, perm = True):
        """ 
    Inputs: 
        
            erp_amplitude: is a dictionary file of two ERPs of each target frequency and phase.
                           This variable is calculated by first averaging over the channels within 
                           the chosen cluster and then averaging over epochs in the main script. 
                           Z scoring happens inside this function. I have one value for each ERP,
                           target freq and target phase and I do z scoring for each ERP, target freq
                           within the phases.
                           If my explanations are not clear enough, below the structure of this variable
                           and how zscoring is done, is shown.
                                                                                ____________                  
                                                                               \  ____0°    \      
                                                            ______ 4Hz ________\ |    .     \       
                                                ______ ERP1|______ 8Hz         \ |    .     \     
                                               |           |         .         \ |____315°  \   
                                               |           |         .         \____________\                     
                               erp_amplitude               |______40Hz               \
                                               |                                     \                                                       
                                               |                                     \
                                               |                                     \                             
                                               |______ ERP2                          \
                                                                                 z scoring
                                                                                     \
                                                                                     \
                                                                                     \
                                                                               cosine fitting
                                                                               
             freq_band:
                 freq_band = np.arange(4,44,4), for real time labels.                                                          
                                                                   """
    
    cosinefit = {}
    amplitudes_cosine = np.zeros([len(freq_band), len(labels)])
    pvalues_cosine = np.zeros([len(freq_band), len(labels)])

    
    
    x = np.radians(np.array([0, 45, 90, 135, 180, 225, 270, 315]))
    #x = phase_bin_means
    
    for i in range(len(erp_amplitude)):
        cosinefit[str(i)] = {}
        for jf, f in enumerate(freq_band):    
            print('cosine fits for frequency {}'.format(f))
            y = zscore(list(erp_amplitude[str(i)][str(f)].values()))
            if(math.isnan(zscore(list(erp_amplitude[str(0)][str(8)].values()))[0]) == True):
                break
           
            cosinefit[str(i)][str(f)] = []
            fits = []
            for phase_start in [-np.pi/2, 0, np.pi/2]:   
        
                amp_start = np.sqrt(np.mean(y**2))
                model = lmfit.Model(cosinus)
                params = model.make_params()
        
                params["amp"].set(value=amp_start, min=0, max=np.ptp(y)/2)
                params["phi"].set(value=phase_start, min=-np.pi, max=np.pi)
                data = {ph: np.mean(y[x == ph]) for ph in np.unique(x)}
                fits.append(model.fit(y, params, x=x))
                
            result = fits[np.argmin([f.aic for f in fits])]
            
            if perm:
                model = lmfit.Model(cosinus)
                params = result.params
                dataperm = []
            
                # use all possible combinations of the 8 phase bins to determine p.
                # Take out the first combination because it is the original
                all_perms = list(itertools.permutations(x))
                del all_perms[0]
            
                for iper in tqdm(range(len(all_perms))):
                    x_shuffled = all_perms[iper]
                    dataperm.append([model,params, y, x_shuffled])
            
                with Pool(4) as p:
                    surrogate = p.starmap(do_one_perm, dataperm)
            else: 
                surrogate = [np.nan]
                
            
            nullmodel = lmfit.Model(unif)
            params = nullmodel.make_params()
            params["offset"].set(value=np.mean(y), min=min(y), max=max(y))
            nullfit = nullmodel.fit(y, params, x=x)
            surrogate = np.array(surrogate)
            surrogate = surrogate[np.invert(np.isnan(surrogate))]
            
            cosinefit[str(i)][str(f)].append( { 'Model': 'cosinus', 
                                    'Frequency': f, 
                                    'Fit': result,
                                    'data': data, 
                                    'amp': result.best_values['amp'], 
                                    'surrogate': surrogate, 
                                    'p':[np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0], 
                                    'std':[np.nan if perm == False else np.std(surrogate)][0], 
                                    'nullmodel':nullfit,
                                    })
            
            amplitudes_cosine[jf, i] = result.best_values['amp']
            pvalues_cosine[jf, i] = [np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0] 
    
    
    
    return cosinefit, amplitudes_cosine, pvalues_cosine



def fig_2a_plot(erp_amplitude, freq_band , subject_info, freq_step_i, save_folder, vmin , vmax):
    # Plotting the heatmap for each ERP Fig2.a Torrecillos 2020
    
    from scipy import ndimage 
    from matplotlib.patches import Rectangle

    def get_largest_component(image):
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
        get the largest component from 2D image
        image: 2d array
        """
        
        # Generate a structuring element that will consider features connected even 
        #s = [[1,0,1],[0,1,0],[1,0,1]] 
        s = ndimage.generate_binary_structure(2,2)
        labeled_array, numpatches = ndimage.label(image, s)
        sizes = ndimage.sum(image, labeled_array, range(1, numpatches + 1))
        max_label = np.where(sizes == sizes.max())[0] + 1
        output = np.asarray(labeled_array == max_label, np.uint8)
        return  output 
        
    

        
    
    
    
        
    for erp in range(len(erp_amplitude)):
        data_erp = {}       
    
        
        for jf, freq in enumerate(freq_band):   
            data_erp[str(freq)] = erp_amplitude[str(erp)][freq]   
        data_erp_df = pd.DataFrame(data_erp) 
        data_erp_df = data_erp_df.T
        data_erp_arr = zscore(data_erp_df.to_numpy())
        fig = plt.figure()
        data_erp_df = pd.DataFrame(data_erp) 
        data_erp_df = data_erp_df.T
        
        
        # Plotting the biggest cluster
        # Setting a threshold to equal to standard deviation of ERP amplitude
        arr = data_erp_arr > np.std(data_erp_arr)
        arr_largest_cluster = get_largest_component(arr)
        
        
        data_erp_df = pd.DataFrame(data_erp) 
        data_erp_df = data_erp_df.T
        data_erp_df_rename = data_erp_df.rename(columns={0:'0', 1:'45', 2:'90', 3:'135', 4:'180', 5:'225', 6:'270', 7:'315'})
        data_erp_df_ph_reorder = data_erp_df_rename.reindex(columns = ["0", "45", "90", "135","180" , "225","270","315"])
        ax = sns.heatmap(data_erp_df_ph_reorder,   cmap ='viridis', vmin = vmin, vmax = vmax)
        arr_largest_cluster_ind = np.argwhere(arr_largest_cluster == 1)
        for i in range(len(arr_largest_cluster_ind)):
            ax.add_patch(Rectangle((arr_largest_cluster_ind[i][1], arr_largest_cluster_ind[i][0]),  1, 1,  ec = 'cyan', fc = 'none', lw=2, hatch='//'))
        
        # swap the axes
        ax.invert_yaxis()
        plt.xlabel("Phases", fontsize=16)
        plt.ylabel("Frequencies", fontsize=16)
        
        if erp==0 and str(subject_info == 'Group Average'):
             print(0)
             plt.title(f'1st ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_biggest_cluster' + '_' + str(subject_info) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
        elif erp==0 and str(subject_info[-2] == 'Experiment'): 
             plt.title(f'1st ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder +  'fig_2c_biggest_cluster' + '_' + str(subject_info[-3]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
        elif erp==1 and str(subject_info == 'Group Average'):
             plt.title(f'2nd ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_biggest_cluster' + '_' + str(subject_info) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
        elif erp==1 and str(subject_info[-2] == 'Experiment'):  
             plt.title(f'2nd ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_biggest_cluster' + '_' + str(subject_info[-3]) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
         
            
         

        plt.figure()
        data_erp_df_rename = data_erp_df.rename(columns={0:'0', 1:'45', 2:'90', 3:'135', 4:'180', 5:'225', 6:'270', 7:'315'})
        data_erp_df_ph_reorder = data_erp_df_rename.reindex(columns = ["0", "45", "90", "135", "180" , "225","270","315"])
        ax = sns.heatmap(data_erp_df_ph_reorder,   cmap ='viridis', vmin = vmin, vmax = vmax)
        arr_all_cluster_ind = np.argwhere(arr == True)
        for i in range(len(arr_all_cluster_ind)):
            ax.add_patch(Rectangle((arr_all_cluster_ind[i][1], arr_all_cluster_ind[i][0]),  1, 1,  ec = 'cyan', fc = 'none', lw=2, hatch='//'))
        
        # swap the axes
        ax.invert_yaxis()
        plt.xlabel("Phases", fontsize=16)
        plt.ylabel("Frequencies", fontsize=16)
        
        if erp==0 and str(subject_info == 'Group Average'):
             plt.title(f'1st ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
        elif erp==0 and str(subject_info[-2] == 'Experiment' ):
             plt.title(f'1st ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
        elif erp==1 and str(subject_info == 'Group Average'):
            plt.title(f'2nd ERP, Subject: {subject_info, freq_step_i}')
            fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
        elif erp==1 and str(subject_info[-2] == 'Experiment'): 
            plt.title(f'2nd ERP, Subject: {subject_info[-3], freq_step_i}')
            fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
            
    return  fig     


def fig_2c_plot(erp_amplitude, freq_band, cosinefit, subject_info, freq_step_i, save_folder):
    # Fig 2.c
    for i in range(len(erp_amplitude)):
        mod_depth = {}
        #data_z = {}
        #data_z_min = {}
        #data_z_max = {}
        best_fit = {}
        best_fit_min = {}
        best_fit_max = {}
        surr_min = {}
        surr_max = {}
        for jf, freq in enumerate(freq_band):   
           mod_depth[str(freq)] = cosinefit[str(i)][str(freq)][0]['amp']
           #data_z[str(freq)] =  np.array(list(cosinefit[str(i)][str(freq)][0]['data'].values()))
           #data_z_min[str(freq)] = min(data_z[str(freq)])
           #data_z_max[str(freq)] = max(data_z[str(freq)])
           best_fit[str(freq)] = cosinefit[str(i)][str(freq)][0]['Fit'].best_fit
           best_fit_min[str(freq)]  = min(cosinefit[str(i)][str(freq)][0]['Fit'].best_fit)
           best_fit_max [str(freq)] = max(cosinefit[str(i)][str(freq)][0]['Fit'].best_fit)
           surr_min[str(freq)] = min(cosinefit[str(i)][str(freq)][0]['surrogate'])
           surr_max[str(freq)] = max(cosinefit[str(i)][str(freq)][0]['surrogate'])
           # Fig2.C
        fig = plt.figure()
        plt.plot(np.array(freq_band), (np.array(list(best_fit_max.values())) + np.array(list(best_fit_min.values())))/1, 'k')
        #plt.fill_between(np.array(freq_band), np.array(list(data_z_min.values())), np.array(list(data_z_max.values())), color = '0.8')
        plt.fill_between(np.array(freq_band), np.array(list(best_fit_min.values())), np.array(list(best_fit_max.values())), color = '0.8')
        #plt.fill_between(np.array(freq_band), np.array(list(surr_min.values())), np.array(list(surr_max.values()))/2, color = 'r')
        #plt.plot(np.array(list(best_fit.values())), 'r')
        plt.xlabel("Frequecies")
        plt.ylabel("Strength of Mod")

            
        if i==0 and str(subject_info == 'Group Average'):
             plt.title(f'1st ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
        elif i==0 and str(subject_info[-2] == 'Experiment' ):
             plt.title(f'1st ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
        elif i==1 and str(subject_info == 'Group Average'):
             plt.title(f'2nd ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + '/cluster_freq/' + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
        elif i==1 and str(subject_info[-2] == 'Experiment'): 
             plt.title(f'2nd ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
    
    return fig







def Select_Epochs(epochs, freq, phase):
    """ 
    this is a function that will identify epochs based on their key (a string) in event_id, 
    which describes the stimulation condition
        
    selection depends on the frequency and the phase of interest
        
    the function returns a list of event indices, that only includes the indices of epochs that contained 
    stimulation at the desired frequency and phase
        
        
    data: epochs data in MNE format
    freq: an integer number, this can be any number between 0 and 40 and depends on the frequencies
    that were stimulated in your study (and thus described in your event description (a string) in event_id)
    phase: an integer number, this can be any number between 0 and 360 and depends on the phases
    that were stimulated in your study (and thus described in your event description in event_id)
    """
    
    index_list = []
    events_array = epochs.events
    event_id_dict = epochs.event_id
    # example o event description for acute NMES study: “freq”: “4”, “phase”: “0”
    freq_to_select = str(freq) 
    phase_to_select = str(phase) 
    
    
    for i in range(len(events_array)):
        event_code = events_array[i,2]
        event_id_key = list(event_id_dict.keys())[list(event_id_dict.values()).index(event_code)]
        
        if freq >= 0 and freq <= 40:
            if phase >= 0 and phase <=360:
                #if (freq_to_select in str(event_id_key[:(event_id_key.find('_') -2)])) == True and (phase_to_select in str(event_id_key[event_id_key.find('_') + 1:])) == True:
                if (freq_to_select == str(event_id_key[:(event_id_key.find('_') -2)]))  and (phase_to_select == str(event_id_key[event_id_key.find('_') + 1:])) :    
                    index_list.append(i)      
                else:
                    continue
            else:
                print("the specified phase is not within the range of 0 to 360 degrees")
        else:
            print("the specified freq is not within the range of 0 to 40 Hz")
    return index_list
    




def clustering_channels():
    
    
    exdir = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
    files = Path(exdir).glob('*.fif*')
    plt.close('all')
    #idx =263
    
    
    labels,_ = ph_analysis.get_ERP_1st_2nd(plot = True)
    
    peaks = np.zeros([27, 64, 2, 8, 10])
    peaks_std =  np.zeros([27, 64, 2, 8, 10])      
    a = np.zeros([64,2])
    target_freq = []
    target_phase = []
    unique_phases = np.arange(0, 360, 45 )
    unique_freqs = np.arange(4, 44, 4)    
    
    for ifiles, f in enumerate(files):
        epochs = mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
        epochs_byfreqandphase = {} 
        erp_byfreqandphase = {} 
        peaks_byfreqandphase = {}
        peaks_byfreqandphase_std = {}
        evoked_zscored  = {}
        
        for ifreq, freq in enumerate(unique_freqs):
            epochs_byfreqandphase[str(freq)] = {}
            erp_byfreqandphase[str(freq)] = {} 
            peaks_byfreqandphase[str(freq)] = {} 
            peaks_byfreqandphase_std[str(freq)] = {}
            evoked_zscored[str(freq)] = {}
            for iphase, phase in enumerate(unique_phases):
                sel_idx = ph_analysis.Select_Epochs(epochs, freq, phase)
                epochs_byfreqandphase[str(freq)][str(phase)] = epochs[sel_idx]
                erp_byfreqandphase[str(freq)][str(phase)]  = epochs_byfreqandphase[str(freq)][str(phase)].average() 
                #peaks_byfreqandphase[str(freq)][str(phase)] = np.mean((erp_byfreqandphase[str(freq)][str(phase)]._data[:,idx:(idx+3)]),1)             
                for ipeak, peak in enumerate(labels):
                    #print(ipeak, peak) 
                    target_phase.append(phase)
                    peaks_byfreqandphase[str(freq)][str(phase)] = np.mean((erp_byfreqandphase[str(freq)][str(phase)]._data[:,peak:(peak + 3)]),1)
                    peaks_byfreqandphase_std[str(freq)][str(phase)] = np.mean((erp_byfreqandphase[str(freq)][str(phase)]._data[:,peak:(peak + 3)]),1)
                    # To remove none arrays after selecting epochs
                    if str(erp_byfreqandphase[str(freq)][str(phase)].comment) == str(''):
                        peaks_byfreqandphase[str(freq)][str(phase)] = np.zeros(64) 
                        peaks_byfreqandphase_std[str(freq)][str(phase)] = np.zeros(64)
                    else:
                        peaks[ifiles, :, ipeak, iphase, ifreq] = peaks_byfreqandphase[str(freq)][str(phase)] 
                        peaks_std[ifiles, :, ipeak, iphase, ifreq] = peaks_byfreqandphase_std[str(freq)][str(phase)]
                    #a[ifiles, :, ipeak] =  peaks[ifiles, :, ipeak] 
    
    
               
                
    unique_phases = np.arange(0, 360, 45 )
    unique_freqs = np.arange(4, 44, 4)        
    
    adjacency_mat,_ = mne.channels.find_ch_adjacency(epochs_byfreqandphase[str(freq)][str(phase)].info , 'eeg')
    clusters, mask, pvals, thresholds = ph_analysis.permutation_cluster(peaks, adjacency_mat)                 
    nsubj, nchans, npeaks, nphas, nfreqs = np.shape(peaks)    
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters)):
        allclusters[:,p] = clusters[p][0]
    # set all other t values to 0 to focus on clusters
    allclusters[mask==False] = 0
    ch_names = epochs.ch_names
    # this is putting the 5-dim data structure in the right format for performing the sine fits
    
    for p in range(len(clusters)):
        a[:,p] = clusters[p][0]
    ph_analysis.plot_topomap_peaks_second_v(a, mask, ch_names, thresholds , pvals,[-5,5])
    
    
    
    # Name and indices of the EEG electrodes that are in the biggest cluster
    all_ch_names_biggets_cluster =  []
    all_ch_ind_biggets_cluster =  []
    
    for p in range(len(clusters)):
        # indices
        all_ch_ind_biggets_cluster.append(np.where(mask[:,p] == 1))
        # channel names
        all_ch_names_biggets_cluster.append([ch_names[i] for i in np.where(mask[:,p] == 1)[0]])
    
    
    all_channels_clustered = all_ch_names_biggets_cluster[0] + all_ch_names_biggets_cluster[1]

        
    return all_channels_clustered, all_ch_names_biggets_cluster[0], all_ch_names_biggets_cluster[1], pvals



def epoch_concat_and_mod_dict(files_GA):
    
    save_folder = "/home/sara/NMES/analyzed data/phase_analysis/"
        
    dict_origin_labels  = {'12Hz_0': 1, '12Hz_135': 2, '12Hz_180': 3, '12Hz_225': 4, '12Hz_270': 5,
                           '12Hz_315': 6, '12Hz_45': 7, '12Hz_90': 8, '16Hz_0': 9, '16Hz_135': 10, 
                           '16Hz_180': 11, '16Hz_225': 12, '16Hz_270': 13, '16Hz_315': 14, '16Hz_45': 15, 
                           '16Hz_90': 16, '20Hz_0': 17, '20Hz_135': 18, '20Hz_180': 19, '20Hz_225': 20, 
                           '20Hz_270': 21, '20Hz_315': 22, '20Hz_45': 23, '20Hz_90': 24, '24Hz_0': 25,
                           '24Hz_135': 26, '24Hz_180': 27, '24Hz_225': 28, '24Hz_270': 29, '24Hz_315': 30, 
                           '24Hz_45': 31, '24Hz_90': 32, '28Hz_0': 33, '28Hz_135': 34, '28Hz_180': 35, 
                           '28Hz_225': 36, '28Hz_270': 37, '28Hz_315': 38, '28Hz_45': 39, '28Hz_90': 40, 
                           '32Hz_0': 41, '32Hz_135': 42, '32Hz_180': 43, '32Hz_225': 44, '32Hz_270': 45, 
                           '32Hz_315': 46, '32Hz_45': 47, '32Hz_90': 48, '36Hz_0': 49, '36Hz_135': 50, 
                           '36Hz_180': 51, '36Hz_225': 52, '36Hz_270': 53, '36Hz_315': 54, '36Hz_45': 55,
                           '36Hz_90': 56, '40Hz_0': 57, '40Hz_135': 58, '40Hz_180': 59, '40Hz_225': 60,
                           '40Hz_270': 61, '40Hz_315': 62, '40Hz_45': 63, '40Hz_90': 64, '4Hz_0': 65,
                           '4Hz_135': 66, '4Hz_180': 67, '4Hz_225': 68, '4Hz_270': 69, '4Hz_315': 70, 
                           '4Hz_45': 71, '4Hz_90': 72, '8Hz_0': 73, '8Hz_135': 74, '8Hz_180': 75,
                           '8Hz_225': 76, '8Hz_270': 77, '8Hz_315': 78, '8Hz_45': 79, '8Hz_90': 80}
    
       
    mod = {}
    all_epochs_list = []
    all_epochs_events = []
    all_names = []
    
    all_channels_clustered,_, _,  pvals = clustering_channels()
    
    for f_GA in files_GA:
        epochs_eeg = mne.read_epochs(f_GA, preload=True)
        # So basically the problem was mne creats a dict of all stimulation conditions in our case 80. For some epochs data with a small
        # size all these 80 conditions are not present. It can be 76 so the dict will start from zero to 76 and event_id keys and value will be 
        # different for each condition in different subjects and there will be a problem during concatinating.
        # I created a diffault dict, based on 80 condition and forced it to be the same for other epoch files even for the one with less
        # than 80 conditions.
        if len(epochs_eeg.event_id) < 80:
            mod_vals = np.zeros([len(epochs_eeg.events[:, 2]), 2])
            #shared_keys = set(epochs_eeg.event_id.keys()).intersection(set(dict_origin_labels))
            for i in epochs_eeg.event_id.keys():
                #print(i)
                mod[i] = [i, epochs_eeg.event_id[str(i)], dict_origin_labels[str(i)]]
            mod_arr = np.array(list(mod.values())) 
            
            for i in range(len(epochs_eeg.events[:, 2])):
                for j in range(len(mod_arr)):
                    if epochs_eeg.events[:, 2][i] == int(mod_arr[j,1]):
                        mod_vals[i] = [mod_arr[j,1],mod_arr[j,2] ]
            epochs_eeg.events[:, 2] = mod_vals[:,1]
            epochs_eeg.event_id = dict_origin_labels
    
        
        all_epochs_list.append(epochs_eeg.pick_channels(['P3', 'P4', 'P7', 'P8', 'Pz', 'CP1', 'P1', 'P2', 'CP3', 'PO3',
               'PO4', 'P5', 'P6', 'POz', 'Cz', 'FC1', 'FC2', 'CP1', 'CP2', 'C1',
               'C2', 'FC3', 'CPz']))
        all_epochs_events.append(epochs_eeg.event_id)
        all_names.append(f_GA.parts[-1][0:9])
    
    all_epochs_concat = mne.concatenate_epochs(all_epochs_list)
    all_epochs_concat.save(save_folder + '/epochs all subjects/' +'epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'bip epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    return all_epochs_concat, all_channels_clustered






def permutation_cluster(peaks, adjacency_mat):
        # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2, -1))
    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    nperm = 100
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([nperm+1, npeaks])
    thresholds=[3, 3]
    pvals = np.zeros([npeaks])
    clusters = []
    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

    for p in range(npeaks):
        cluster = mne.stats.permutation_cluster_1samp_test(mean_peaks[:,:,p], out_type='mask',
                                                           adjacency=adjacency_mat, threshold=thresholds[p],
                                                           n_permutations=1000)
        t_sum = np.zeros([len(cluster[1])])
        # get the sum of the tvalues for each of the 
        # clusters to choose the main cluster 
        # (take magnitude to treat negative and positive cluster equally)
        for c in range(len(cluster[1])):
            t_sum[c] = np.abs(sum(cluster[0][cluster[1][c]]))
    
        # store the maximal cluster size for each iteration 
        # to later calculate p value
        # if no cluster was found, put in 0
        if len(t_sum) > 0:
            max_cluster_size[p] = np.max(t_sum)

            mask[:,p] = cluster[1][np.argmax(t_sum)]
            
            # in this part I have chosen negative clusters.
            if p==0:
                mask[:,p] = cluster[1][1]
                pvals[0] = cluster[2][1]
            else:
                mask[:,p] = cluster[1][2]
                pvals[1] = cluster[2][2]
                
                
        clusters.append(cluster)         
        

    

    return clusters, mask, pvals, thresholds





def plot_topomap_peaks_second_v(peaks, mask, ch_names, thresholds, pvals, clim):

    import matplotlib.pyplot as plt

    nplots =1 
    nchans, npeaks = np.shape(peaks)

    maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k',
                linewidth=0, markersize=5)

    fig, sps = plt.subplots(nrows=nplots, ncols=npeaks)
    
    
    for iplot in range(nplots):
        for ipeak in range(npeaks):

            # if mask is None:
            #     psig = None
            # else:
            #     psig = np.where(mask[iplot, :, ipeak] < 0.01, True, False)

            # sps[ipeak, iplot].set_aspect('equal')

            if mask is not None:
                imask=mask[:,ipeak]
            else:
                imask = None

            im = topoplot_2d (  ch_names, peaks[ :, ipeak], 
                                clim=clim, axes=sps[ipeak], 
                                mask=imask, maskparam=maskparam)

    fig.subplots_adjust(wspace=0, hspace=0)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    
    fig.suptitle('All Frequencies and Peak and trough Phases Vs zero', fontsize = 14)
    #fig.suptitle('All Frequencies and all phases', fontsize = 14)
    sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n  cluster_pv = {pvals[0]}')
    sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = {pvals[1]}')
    cb.set_label('t-value', rotation = 90)

    

    plt.show()

    return im




def topoplot_2d (ch_names, ch_attribute, clim=None, axes=None, mask=None, maskparam=None):
    
    """
    Function to plot the EEG channels in a 2d topographical plot by color coding 
    a certain attribute of the channels (such as PSD, channel specific r-squared).
    Draws headplot and color fields.
    Parameters
    ----------
    ch_names : String of channel names to plot.
    ch_attribute : vector of values to be color coded, with the same length as the channel, numerical.
    clim : 2-element sequence with minimal and maximal value of the color range.
           The default is None.
           
    Returns
    -------
    None.
    This function is a modified version of viz.py (mkeute, github)
    """    

    import mne
    # get standard layout with over 300 channels
    layout = mne.channels.read_layout('EEG1005')
    
    # select the channel positions with the specified channel names
    # channel positions need to be transposed so that they fit into the headplot
    pos = (np.asanyarray([layout.pos[layout.names.index(ch)] for ch in ch_names])
           [:, 0:2] - 0.5) / 5
    
    if maskparam == None:
        maskparam = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                    linewidth=0, markersize=2) #default in mne
    if clim == None:
        im = mne.viz.plot_topomap(ch_attribute, 
                                  pos, 
                                  cmap = 'jet',
                                  axes=axes,
                                  outlines = "head",
                                  mask=mask,
                                  mask_params=maskparam,
                                  vmin = np.min(ch_attribute),
                                  vmax = np.max(ch_attribute))
    else:
        im = mne.viz.plot_topomap(ch_attribute, 
                                  pos, 
                                  sensors=False,
                                  contours=3,
                                  cmap = 'jet',
                                  axes=axes,
                                  outlines = "head", 
                                  mask=mask,
                                  mask_params=maskparam,
                                  vmin = clim[0], 
                                  vmax = clim[1])
    return im





import lmfit
from tqdm import tqdm
from multiprocessing import Pool
import itertools 
from scipy.interpolate import interp1d




def cosinus(x, amp, phi):
    return amp * np.cos(x + phi)

def unif(x, offset):
    return offset

def do_one_perm(model, params, y,x):
    resultperm = model.fit(y, params, x=x)
    return resultperm.best_values['amp']




def do_cosine_fit_phase_freq_extracted(erp_amplitude, erp_amplitude_sem, phase_bin_means, freq_band, labels, perm = True):
    '''Args:
           erp_amplitude:
               dict variable that contains amplitude of first and second peak
           freq_band:
               we fit the cosine to freq_band = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40] (freq_step_i == '4Hz step')
               to be comparable with Lucky loop labels.
           labels: 
               time points of first and second erp peaks estimated by GFP     
               
           
        '''
        
        
        
    
    cosinefit = {}
    amplitudes_cosine = np.zeros([len(freq_band), len(labels)])
    pvalues_cosine = np.zeros([len(freq_band), len(labels)])



    x = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    #x = phase_bin_means
    
    for i in range(len(erp_amplitude)):
        fig, sps = plt.subplots(nrows=2, ncols=int(len(freq_band)/2))
        if i == 0:
            fig.suptitle('ERP 1')
        else:
            fig.suptitle('ERP 2')
        
        
        cosinefit[str(i)] = {}
        for jf, f in enumerate(freq_band):    
            print('cosine fits for frequency {}'.format(f))
            y = zscore(np.array(list(erp_amplitude[str(i)][f].values())))  
            y_sem = np.array(list(erp_amplitude_sem[str(i)][f].values()))
            cosinefit[str(i)][str(f)] = []
            fits = []
            for phase_start in [-np.pi/2, 0, np.pi/2]:      
        
                amp_start = np.sqrt(np.mean(y**2))
                model = lmfit.Model(cosinus)
                params = model.make_params()
        
                params["amp"].set(value=amp_start, min=0, max=np.ptp(y)/2)
                params["phi"].set(value=phase_start, min=-np.pi, max=np.pi)
                data = {ph: np.mean(y[x == ph]) for ph in np.unique(x)}
                fits.append(model.fit(y, params, x=x))
                
            result = fits[np.argmin([f.aic for f in fits])]
            
            if perm:
                model = lmfit.Model(cosinus)
                params = result.params
                dataperm = []
            
                # use all possible combinations of the 8 phase bins to determine p.
                # Take out the first combination because it is the original
                all_perms = list(itertools.permutations(x))
                del all_perms[0]
            
                for iper in tqdm(range(len(all_perms))):
                    x_shuffled = all_perms[iper]
                    dataperm.append([model,params, y, x_shuffled])
            
                with Pool(4) as p:
                    surrogate = p.starmap(do_one_perm, dataperm)
            else: 
                surrogate = [np.nan]
                
            
            nullmodel = lmfit.Model(unif)
            params = nullmodel.make_params()
            params["offset"].set(value=np.mean(y), min=min(y), max=max(y))
            nullfit = nullmodel.fit(y, params, x=x)
            surrogate = np.array(surrogate)
            surrogate = surrogate[np.invert(np.isnan(surrogate))]
            
            cosinefit[str(i)][str(f)].append( { 'Model': 'cosinus', 
                                    'Frequency': f, 
                                    'Fit': result,
                                    'data': data, 
                                    'amp': result.best_values['amp'], 
                                    'surrogate': surrogate, 
                                    'p':[np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0], 
                                    'std':[np.nan if perm == False else np.std(surrogate)][0], 
                                    'nullmodel':nullfit,
                                    })
            
            amplitudes_cosine[jf, i] = result.best_values['amp']
            pvalues_cosine[jf, i] = [np.nan if perm == False else sum(np.abs(surrogate) >= np.abs(result.best_values['amp']))/len(surrogate)][0] 
    
            if jf == 0:
                row = 0; col=0
            elif jf == 1:
                row = 0; col=1
            elif jf == 2:
                row = 0; col=2   
            elif jf == 3:
                row = 0; col=3
            elif jf == 4:
                row = 0; col=4 
            elif jf == 5:
                row = 1; col=0
            elif jf == 6:
                row = 1; col=1   
            elif jf == 7:
                row = 1; col=2
            elif jf == 8:
                row = 1; col=3 
            elif jf == 9:
                row = 1; col=4            
                
                 
            
            sps[row, col].errorbar(x, y, fmt='.k')
            sps[row, col].errorbar(x, y, yerr = y_sem, fmt='.k')
            sps[row, col].set_title(f'{int(f)} Hz')
            cosin = cosinus(x, result.values['amp'], result.values['phi'])
            #sps[row, col].plot(unique_phases, cosin, 'r')
            xnew = np.linspace(x[0], x[-1], num=41, endpoint=True)
            f2 = interp1d(x, cosin, kind='cubic')
            sps[row, col].plot(xnew, f2(xnew), 'r', label = 'Fitted Cosine')
            sps[row, col].plot([0,np.max(x)], [0,0], 'k', label = 'Data')
            sps[row, col].set_ylim([-5,5])
            sps[row, col].set_xlim([-5,320])
            sps[row, col].grid()
            sps[row, col].legend()
        
    
            if pvalues_cosine[jf, i]<0.05:
                sps[row, col].text(0.5,0.5, str(round(float(pvalues_cosine[jf, i]),4)), c='r')
            else:
                sps[row, col].text(0.5,0.5, str(round(float(pvalues_cosine[jf, i]),4)), c='k')
    
            plt.show()

    
    return cosinefit, amplitudes_cosine, pvalues_cosine









def epoch_concat_and_mod_dict_bip(files_GA):
    
    save_folder = "/home/sara/NMES/analyzed data/phase_analysis/"
        
    dict_origin_labels  = {'12Hz_0': 1, '12Hz_135': 2, '12Hz_180': 3, '12Hz_225': 4, '12Hz_270': 5,
                           '12Hz_315': 6, '12Hz_45': 7, '12Hz_90': 8, '16Hz_0': 9, '16Hz_135': 10, 
                           '16Hz_180': 11, '16Hz_225': 12, '16Hz_270': 13, '16Hz_315': 14, '16Hz_45': 15, 
                           '16Hz_90': 16, '20Hz_0': 17, '20Hz_135': 18, '20Hz_180': 19, '20Hz_225': 20, 
                           '20Hz_270': 21, '20Hz_315': 22, '20Hz_45': 23, '20Hz_90': 24, '24Hz_0': 25,
                           '24Hz_135': 26, '24Hz_180': 27, '24Hz_225': 28, '24Hz_270': 29, '24Hz_315': 30, 
                           '24Hz_45': 31, '24Hz_90': 32, '28Hz_0': 33, '28Hz_135': 34, '28Hz_180': 35, 
                           '28Hz_225': 36, '28Hz_270': 37, '28Hz_315': 38, '28Hz_45': 39, '28Hz_90': 40, 
                           '32Hz_0': 41, '32Hz_135': 42, '32Hz_180': 43, '32Hz_225': 44, '32Hz_270': 45, 
                           '32Hz_315': 46, '32Hz_45': 47, '32Hz_90': 48, '36Hz_0': 49, '36Hz_135': 50, 
                           '36Hz_180': 51, '36Hz_225': 52, '36Hz_270': 53, '36Hz_315': 54, '36Hz_45': 55,
                           '36Hz_90': 56, '40Hz_0': 57, '40Hz_135': 58, '40Hz_180': 59, '40Hz_225': 60,
                           '40Hz_270': 61, '40Hz_315': 62, '40Hz_45': 63, '40Hz_90': 64, '4Hz_0': 65,
                           '4Hz_135': 66, '4Hz_180': 67, '4Hz_225': 68, '4Hz_270': 69, '4Hz_315': 70, 
                           '4Hz_45': 71, '4Hz_90': 72, '8Hz_0': 73, '8Hz_135': 74, '8Hz_180': 75,
                           '8Hz_225': 76, '8Hz_270': 77, '8Hz_315': 78, '8Hz_45': 79, '8Hz_90': 80}
    
    
    
    
    
    
    
       
    mod = {}
    all_epochs_list = []
    all_epochs_events = []
    all_names = []
    
    for f_GA in files_GA:
        epochs_eeg = mne.read_epochs(f_GA, preload=True)
        # So basically the problem was mne creats a dict of all stimulation conditions in our case 80. For some epochs data with a small
        # size all these 80 conditions are not present. It can be 76 so the dict will start from zero to 76 and event_id keys and value will be 
        # different for each condition in different subjects and there will be a problem during concatinating.
        # I created a diffault dict, based on 80 condition and forced it to be the same for other epoch files even for the one with less
        # than 80 conditions.
        if len(epochs_eeg.event_id) < 80:
            mod_vals = np.zeros([len(epochs_eeg.events[:, 2]), 2])
            #shared_keys = set(epochs_eeg.event_id.keys()).intersection(set(dict_origin_labels))
            for i in epochs_eeg.event_id.keys():
                #print(i)
                mod[i] = [i, epochs_eeg.event_id[str(i)], dict_origin_labels[str(i)]]
            mod_arr = np.array(list(mod.values())) 
            
            for i in range(len(epochs_eeg.events[:, 2])):
                for j in range(len(mod_arr)):
                    if epochs_eeg.events[:, 2][i] == int(mod_arr[j,1]):
                        mod_vals[i] = [mod_arr[j,1],mod_arr[j,2] ]
            epochs_eeg.events[:, 2] = mod_vals[:,1]
            epochs_eeg.event_id = dict_origin_labels
    
        # channels based on clustered channels. Only using those because the size of this variable will be very large. 
        all_epochs_list.append(epochs_eeg)
        all_epochs_events.append(epochs_eeg.event_id)
        all_names.append(f_GA.parts[-1][0:9])
    
    all_epochs_concat = mne.concatenate_epochs(all_epochs_list)
    #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'bip epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    return all_epochs_concat











# =============================================================================
# 
# 
# 
# 
# def epoch_concat_and_mod_dict_bip(files_GA):
#     
#     save_folder = "/home/sara/NMES/analyzed data/phase_analysis/"
#     
#     
#     
#     
#     
#     
#     
#     
#        
#     mod = {}
#     all_epochs_list = []
#     all_epochs_events = []
#     all_names = []
#     
#     for f_GA in files_GA:
#         epochs_eeg = mne.read_epochs(f_GA, preload=True)
# 
#     
#         # channels based on clustered channels. Only using those because the size of this variable will be very large. 
#         all_epochs_list.append(epochs_eeg)
#         all_epochs_events.append(epochs_eeg.event_id)
#         all_names.append(f_GA.parts[-1][0:9])
#     
#     all_epochs_concat = mne.concatenate_epochs(all_epochs_list)
#     #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
#     #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'bip epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
#     return all_epochs_concat
# 
# 
# 
# 
# =============================================================================




def epoch_concat_clustered_and_mod_dict(files_GA):
    
    save_folder = "/home/sara/NMES/analyzed data/phase_analysis/"
        
    dict_origin_labels  = {'12Hz_0': 1, '12Hz_135': 2, '12Hz_180': 3, '12Hz_225': 4, '12Hz_270': 5,
                           '12Hz_315': 6, '12Hz_45': 7, '12Hz_90': 8, '16Hz_0': 9, '16Hz_135': 10, 
                           '16Hz_180': 11, '16Hz_225': 12, '16Hz_270': 13, '16Hz_315': 14, '16Hz_45': 15, 
                           '16Hz_90': 16, '20Hz_0': 17, '20Hz_135': 18, '20Hz_180': 19, '20Hz_225': 20, 
                           '20Hz_270': 21, '20Hz_315': 22, '20Hz_45': 23, '20Hz_90': 24, '24Hz_0': 25,
                           '24Hz_135': 26, '24Hz_180': 27, '24Hz_225': 28, '24Hz_270': 29, '24Hz_315': 30, 
                           '24Hz_45': 31, '24Hz_90': 32, '28Hz_0': 33, '28Hz_135': 34, '28Hz_180': 35, 
                           '28Hz_225': 36, '28Hz_270': 37, '28Hz_315': 38, '28Hz_45': 39, '28Hz_90': 40, 
                           '32Hz_0': 41, '32Hz_135': 42, '32Hz_180': 43, '32Hz_225': 44, '32Hz_270': 45, 
                           '32Hz_315': 46, '32Hz_45': 47, '32Hz_90': 48, '36Hz_0': 49, '36Hz_135': 50, 
                           '36Hz_180': 51, '36Hz_225': 52, '36Hz_270': 53, '36Hz_315': 54, '36Hz_45': 55,
                           '36Hz_90': 56, '40Hz_0': 57, '40Hz_135': 58, '40Hz_180': 59, '40Hz_225': 60,
                           '40Hz_270': 61, '40Hz_315': 62, '40Hz_45': 63, '40Hz_90': 64, '4Hz_0': 65,
                           '4Hz_135': 66, '4Hz_180': 67, '4Hz_225': 68, '4Hz_270': 69, '4Hz_315': 70, 
                           '4Hz_45': 71, '4Hz_90': 72, '8Hz_0': 73, '8Hz_135': 74, '8Hz_180': 75,
                           '8Hz_225': 76, '8Hz_270': 77, '8Hz_315': 78, '8Hz_45': 79, '8Hz_90': 80}
    
    
    
    # These lines go to the permutation cluster function and select the channels that will be appended 
    # in the epoch list.
    
    
    all_channels_clustered, ERP_1_ch, ERP_2_ch, pvals = clustering_channels()
       
    mod = {}
    all_epochs_list = []
    all_epochs_events = []
    all_names = []
    
    for f_GA in files_GA:
        epochs_eeg = mne.read_epochs(f_GA, preload=True)
        # So basically the problem was mne creats a dict of all stimulation conditions in our case 80. For some epochs data with a small
        # size all these 80 conditions are not present. It can be 76 so the dict will start from zero to 76 and event_id keys and value will be 
        # different for each condition in different subjects and there will be a problem during concatinating.
        # I created a diffault dict, based on 80 condition and forced it to be the same for other epoch files even for the one with less
        # than 80 conditions.
        if len(epochs_eeg.event_id) < 80:
            mod_vals = np.zeros([len(epochs_eeg.events[:, 2]), 2])
            #shared_keys = set(epochs_eeg.event_id.keys()).intersection(set(dict_origin_labels))
            for i in epochs_eeg.event_id.keys():
                #print(i)
                mod[i] = [i, epochs_eeg.event_id[str(i)], dict_origin_labels[str(i)]]
            mod_arr = np.array(list(mod.values())) 
            
            for i in range(len(epochs_eeg.events[:, 2])):
                for j in range(len(mod_arr)):
                    if epochs_eeg.events[:, 2][i] == int(mod_arr[j,1]):
                        mod_vals[i] = [mod_arr[j,1],mod_arr[j,2] ]
            epochs_eeg.events[:, 2] = mod_vals[:,1]
            epochs_eeg.event_id = dict_origin_labels
    
        # channels based on clustered channels. Only using those because the size of this variable will be very large. 
        all_epochs_list.append(epochs_eeg.pick_channels(all_channels_clustered))
        all_epochs_events.append(epochs_eeg.event_id)
        all_names.append(f_GA.parts[-1][0:9])
    
    all_epochs_concat = mne.concatenate_epochs(all_epochs_list)
    #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'bip epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    return all_epochs_concat, all_channels_clustered, ERP_1_ch, ERP_2_ch




def epoch_concat_subs_mutltiple_files(files_GA):
    
    save_folder = "/home/sara/NMES/analyzed data/phase_analysis/"
        
    dict_origin_labels  = {'12Hz_0': 1, '12Hz_135': 2, '12Hz_180': 3, '12Hz_225': 4, '12Hz_270': 5,
                           '12Hz_315': 6, '12Hz_45': 7, '12Hz_90': 8, '16Hz_0': 9, '16Hz_135': 10, 
                           '16Hz_180': 11, '16Hz_225': 12, '16Hz_270': 13, '16Hz_315': 14, '16Hz_45': 15, 
                           '16Hz_90': 16, '20Hz_0': 17, '20Hz_135': 18, '20Hz_180': 19, '20Hz_225': 20, 
                           '20Hz_270': 21, '20Hz_315': 22, '20Hz_45': 23, '20Hz_90': 24, '24Hz_0': 25,
                           '24Hz_135': 26, '24Hz_180': 27, '24Hz_225': 28, '24Hz_270': 29, '24Hz_315': 30, 
                           '24Hz_45': 31, '24Hz_90': 32, '28Hz_0': 33, '28Hz_135': 34, '28Hz_180': 35, 
                           '28Hz_225': 36, '28Hz_270': 37, '28Hz_315': 38, '28Hz_45': 39, '28Hz_90': 40, 
                           '32Hz_0': 41, '32Hz_135': 42, '32Hz_180': 43, '32Hz_225': 44, '32Hz_270': 45, 
                           '32Hz_315': 46, '32Hz_45': 47, '32Hz_90': 48, '36Hz_0': 49, '36Hz_135': 50, 
                           '36Hz_180': 51, '36Hz_225': 52, '36Hz_270': 53, '36Hz_315': 54, '36Hz_45': 55,
                           '36Hz_90': 56, '40Hz_0': 57, '40Hz_135': 58, '40Hz_180': 59, '40Hz_225': 60,
                           '40Hz_270': 61, '40Hz_315': 62, '40Hz_45': 63, '40Hz_90': 64, '4Hz_0': 65,
                           '4Hz_135': 66, '4Hz_180': 67, '4Hz_225': 68, '4Hz_270': 69, '4Hz_315': 70, 
                           '4Hz_45': 71, '4Hz_90': 72, '8Hz_0': 73, '8Hz_135': 74, '8Hz_180': 75,
                           '8Hz_225': 76, '8Hz_270': 77, '8Hz_315': 78, '8Hz_45': 79, '8Hz_90': 80}
    
    
    
    # These lines go to the permutation cluster function and select the channels that will be appended 
    # in the epoch list.
    

    mod = {}
    all_epochs_list = []
    all_epochs_events = []
    all_names = []
    
    for f_GA in files_GA:
        epochs_eeg = mne.read_epochs(f_GA, preload=True)
        # So basically the problem was mne creats a dict of all stimulation conditions in our case 80. For some epochs data with a small
        # size all these 80 conditions are not present. It can be 76 so the dict will start from zero to 76 and event_id keys and value will be 
        # different for each condition in different subjects and there will be a problem during concatinating.
        # I created a diffault dict, based on 80 condition and forced it to be the same for other epoch files even for the one with less
        # than 80 conditions.
        if len(epochs_eeg.event_id) < 80:
            mod_vals = np.zeros([len(epochs_eeg.events[:, 2]), 2])
            #shared_keys = set(epochs_eeg.event_id.keys()).intersection(set(dict_origin_labels))
            for i in epochs_eeg.event_id.keys():
                #print(i)
                mod[i] = [i, epochs_eeg.event_id[str(i)], dict_origin_labels[str(i)]]
            mod_arr = np.array(list(mod.values())) 
            
            for i in range(len(epochs_eeg.events[:, 2])):
                for j in range(len(mod_arr)):
                    if epochs_eeg.events[:, 2][i] == int(mod_arr[j,1]):
                        mod_vals[i] = [mod_arr[j,1],mod_arr[j,2] ]
            epochs_eeg.events[:, 2] = mod_vals[:,1]
            epochs_eeg.event_id = dict_origin_labels
    
        # channels based on clustered channels. Only using those because the size of this variable will be very large. 
        all_epochs_list.append(epochs_eeg)
        all_epochs_events.append(epochs_eeg.event_id)
        all_names.append(f_GA.parts[-1][0:9])
    
    all_epochs_concat = mne.concatenate_epochs(all_epochs_list)
    #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'bip epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
    return all_epochs_concat, f_GA.parts


# =============================================================================
# 
# 
# 
# 
# 
# 
# 
# def epoch_concat_clustered_and_mod_dict(files_GA):
#     
#     save_folder = "/home/sara/NMES/analyzed data/phase_analysis/"
#         
# 
#     
#     # These lines go to the permutation cluster function and select the channels that will be appended 
#     # in the epoch list.
#     
#     
#     all_channels_clustered, ERP_1_ch, ERP_2_ch, pvals = clustering_channels()
#        
#     mod = {}
#     all_epochs_list = []
#     all_epochs_events = []
#     all_names = []
#     
#     for f_GA in files_GA:
#         epochs_eeg = mne.read_epochs(f_GA, preload=True)
#         # So basically the problem was mne creats a dict of all stimulation conditions in our case 80. For some epochs data with a small
#         # size all these 80 conditions are not present. It can be 76 so the dict will start from zero to 76 and event_id keys and value will be 
#         # different for each condition in different subjects and there will be a problem during concatinating.
#         # I created a diffault dict, based on 80 condition and forced it to be the same for other epoch files even for the one with less
#         # than 80 conditions.
# 
#     
#         # channels based on clustered channels. Only using those because the size of this variable will be very large. 
#         all_epochs_list.append(epochs_eeg.pick_channels(all_channels_clustered))
#         all_epochs_events.append(epochs_eeg.event_id)
#         all_names.append(f_GA.parts[-1][0:9])
#     
#     all_epochs_concat = mne.concatenate_epochs(all_epochs_list)
#     #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
#     #all_epochs_concat.save(save_folder + '/epochs all subjects/' +'bip epochs_all_sub_epo.fif'  , overwrite = True, split_size='2GB')
#     return all_epochs_concat, all_channels_clustered, ERP_1_ch, ERP_2_ch
# 
# =============================================================================














def phase_optimal_distribution(ax, mag_df_array, title):
    
    bins_number = 8 
    bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
    bin_phases= np.array([0, 45, 90, 135, 180, 225, 270, 315])

    width = 2 * np.pi / bins_number
    ax.bar(bins[:bins_number], abs(mag_df_array), zorder=1, align='edge', width=width,  edgecolor='C0', fill=False, linewidth=1)
    mag_all = []
    for j, j_mag in enumerate(mag_df_array):
        #print(bin_phases[j], abs(j_mag))
        mag_all.append( P2R(abs(j_mag), bin_phases[j]))
    
    r, theta = R2P(np.mean(mag_all))
    ax.plot([0, np.degrees((theta))], [0, r],  lw=3, color = 'red')    
    #ax.set_ylim([0,0.6])
    ax.set_title(title,fontweight="bold")  

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return abs(x), np.angle(x)
