#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:45:43 2022

@author: sara
"""





def epoch_concat_and_mod_dict(files_GA):
    import mne
    
        
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
    
    
    #all_channels_clustered, ERP_1_ch, ERP_2_ch = clustering_channels()
    labels, _ = ph_analysis.get_ERP_1st_2nd(plot = True) 
    
    
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
    return all_epochs_concat, labels










def permutation_cluster(peaks, adjacency_mat, thresholds):
        # in this function, peaks is a 5 dim matrix with dims, phases [0,180], nsubj, erps, nfreq, nch
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (0, -2))
    # get matrix dimensions
    nsubj, npeaks, nchans = np.shape(mean_peaks)
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([npeaks])
    thresholds=thresholds
    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster


    for p in range(npeaks):
        cluster = mne.stats.permutation_cluster_1samp_test(mean_peaks[:,p,:], out_type='mask',
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
            # save the original cluster information (1st iteration) 


                # get the channels which are in the main cluster
            mask[:,p] = cluster[1][np.argmax(t_sum)]
        else:
            max_cluster_size[p] = 0
            
            
        clusters.append(cluster)   
    return clusters, mask   









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






def plot_topomap_peaks_second_v(peaks_tval, mask, ch_names, clim):

    import matplotlib.pyplot as plt

    nplots =1 
    nchans, npeaks = np.shape(peaks_tval)

    maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k',
                linewidth=0, markersize=5)

    fig, sps = plt.subplots(nrows=nplots, ncols=npeaks, figsize=(8, 6))
    
    
    
    for iplot in range(nplots):
        for ipeak in range(npeaks):

            if mask is not None:
                imask=mask[:,ipeak]
            else:
                imask = None

            im = topoplot_2d (  ch_names, peaks_tval[ :, ipeak], 
                                clim=clim, axes=sps[ipeak], 
                                mask=imask, maskparam=maskparam)

    fig.subplots_adjust(wspace=0, hspace=0)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
    cb.ax.tick_params(labelsize=12)
    
                 

    
    plt.show()
    return fig, sps, cb















def permutation_cluster_peak_vs_trough_new(peaks, adjacency_mat, thresholds, freq_band):
        # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    
    if freq_band == 'all':
    
        mean_peaks_freq = np.mean(peaks, ( -2))
        
    else:
        
        mean_peaks_freq = peaks
    
    #first row : peak - trough
    # According to mne "The first dimension should correspond to the difference between paired samples (observations) in two conditions. "

    # get matrix dimensions
    nsubj = 27; nchans = 64; npeaks= 2 
    
    nperm = 100
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([nperm+1, npeaks])
    thresholds=thresholds
    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

   
    for p in range(npeaks):
        mean_peaks_phase = mean_peaks_freq[0, :, p, :] -  mean_peaks_freq[1, :, p, :]
        cluster = mne.stats.permutation_cluster_1samp_test(mean_peaks_phase, out_type='mask',
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
            max_cluster_size[ p] = np.max(t_sum)
   
                # get the channels which are in the main cluster
            mask[:,p] = cluster[1][np.argmax(t_sum)]
        else:
            max_cluster_size[ p] = 0

        clusters.append(cluster)

    return clusters, mask








def permutation_cluster_peak_vs_trough_new_each(peaks, adjacency_mat, thresholds, freq_band):
        # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    
    if freq_band == 'all':
    
        mean_peaks_freq = np.mean(peaks, ( -2))
        
    else:
        
        mean_peaks_freq = peaks
    
    #first row : peak - trough
    # According to mne "The first dimension should correspond to the difference between paired samples (observations) in two conditions. "

    # get matrix dimensions
    nsubj = 27; nchans = 64; npeaks= 2 
    
    nperm = 100
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([nperm+1, npeaks])
    thresholds=thresholds
    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

   
    for p in range(npeaks):
        mean_peaks_phase = mean_peaks_freq[0, :, p, :] -  mean_peaks_freq[1, :, p, :]
        cluster = mne.stats.permutation_cluster_1samp_test(mean_peaks_phase, out_type='mask',
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
            max_cluster_size[ p] = np.max(t_sum)
   
                # get the channels which are in the main cluster
            mask[:,p] = cluster[1][np.argmax(t_sum)]
        else:
            max_cluster_size[ p] = 0

        clusters.append(cluster)

    return clusters, mask






def plot_retro_peak_vs_trough_1hz(clusters, ch_names, mask, peak, cluster_pv_freq):

    p = peak
    
    maskparam=None
    
    fig, axes = plt.subplots(nrows=4, ncols=10, figsize=(20,8))
    if peak == 0:
        erp = 'ERP 1'
    else:
        erp = 'ERP 2 '
    fig.suptitle(f'Retrospective Analysis, 1 Hz Step, {erp} ', fontsize = 18, fontweight="bold")
    
    cols = [format(col) for col in np.arange(1,11,1)]
    for ax, col in zip(axes[0], cols):
        ax.set_title(f'{col} Hz', fontsize = 18, fontweight="bold")   
        im = topoplot_2d (  ch_names, clusters[str(col)][p][0], 
                                     clim=[-5,5], axes=ax, 
                                     mask=mask[str(col)][:,p], maskparam=maskparam)
        fig.subplots_adjust(wspace=0, hspace=0)
        cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    
    
    
    cols = [format(col) for col in np.arange(11,21,1)]
    for ax, col in zip(axes[1], cols):
        ax.set_title(f'{col} Hz', fontsize = 18, fontweight="bold")   
        im = topoplot_2d (  ch_names, clusters[str(col)][p][0], 
                                     clim=[-5,5], axes=ax, 
                                     mask=mask[str(col)][:,p], maskparam=maskparam)
        fig.subplots_adjust(wspace=0, hspace=0)
        cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    
    
    cols = [format(col) for col in np.arange(21,31,1)]
    for ax, col in zip(axes[2], cols):
        ax.set_title(f'{col} Hz', fontsize = 18, fontweight="bold")   
        im = topoplot_2d (  ch_names, clusters[str(col)][p][0], 
                                     clim=[-5,5], axes=ax, 
                                     mask=mask[str(col)][:,p], maskparam=maskparam)
        fig.subplots_adjust(wspace=0, hspace=0)
        cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    
    
    cols = [format(col) for col in np.arange(31,41,1)]
    for ax, col in zip(axes[3], cols):
        ax.set_title(f'{col} Hz', fontsize = 18, fontweight="bold")   
        im = topoplot_2d (  ch_names, clusters[str(col)][p][0], 
                                     clim=[-5,5], axes=ax, 
                                     mask=mask[str(col)][:,p], maskparam=maskparam)
        fig.subplots_adjust(wspace=0, hspace=0)
        cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    
    
    for i_l, l in enumerate(np.where(cluster_pv_freq[:,p]>0)[0]):
    
        if l < 10:
            row = 0; col =l
        elif l > 10 and l < 20:
            row = 1; col = l -10
        elif l >20 and l< 30:
            row = 2; col = l - 20
        elif l >30 and l< 40:
            row = 3; col = l - 30
            
            
        axes[row, (col-1)].set_xlabel(f' P = {cluster_pv_freq[l, p]}', fontsize = 18, fontweight="bold")
    plt.tight_layout()

    
    return fig
    
#%%
import os
import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

# phase analysis function
import phase_analysis_function as ph_analysis
#%% Extracting phase and frequency from the bipolar channel around C3.

exdir = "/home/sara/NMES/NMES_Experimnet/"
files = list(Path(exdir).glob('**/*.xdf'))
#save_folder =  "/home/sara/NMES/analyzed data/phase_analysis"
#save_folder_pickle =  "/home/sara/NMES/analyzed data/phase_analysis/pickle/"
#save_folder_fig = "/home/sara/NMES/analyzed data/phase_analysis/Figures/cluster_freq/each_subject"



bin_num = 8 # So it will be consistent with online target phases
freq_steps = ['1Hz step', '4Hz step']
peaks_freq = np.zeros([27, 2, 2, 41, 8, 64])


for i_files, f in enumerate(files):

    plt.close('all')
    # Step 1: reading XDF files and changing their format to raw mne just for the bipolar channel around C3.
    # We chose this channel because real-time analysis for labeling the stimulation is done according to this channel. 
    raw_bip, subject_info = ph_analysis.XDF_correct_time_stamp_reject_pulses_bip(f)
    
    
    # Step 2: Create events from the annotations present in the raw file
    # excluding non-unique events and time-stamps
    (events_from_annot, event_dict) = mne.events_from_annotations(raw_bip)
    u, indices = np.unique(events_from_annot[:,0], return_index=True)
    events_from_annot_unique = events_from_annot[indices]
    # Create epochs based on the events, from -1 to 0s. For extracting phase and frequency, no preprocessing steps are needed. So, 
    # the epochs are made from the raw files before the stimulation. 
    epochs_bip = mne.Epochs(raw_bip, events_from_annot_unique, event_id = event_dict, tmin=-1, tmax= 0, reject=None, preload=True)
    #epochs_bip.save(save_folder +'/epochs bipolar/' + 'epochs_'+ str(f.parts[-3] +'_' + str(f.parts[-1][39:43] )  ) +'_epo.fif' , overwrite = True, split_size='2GB')

    # Step 3: extractinf phase and frequency. Functions in this step are written by Johanna.
    # An array, rows: frequencies of interests. columns: number of epochs. values in this array are phases.


    
    for freq_step_i, freq_step in enumerate(freq_steps):
        print(freq_step_i, freq_step)
        epochs_bip_fil_phase, target_freq = ph_analysis.extract_phase(epochs_bip._data[: , 1, :], 1000, freq_step, 2, angletype = 'degree')
        #(frequencies of interests, number of epochs). An array like eeg_epoch_filt_phase, but the values are the classes of the phse bins.
        # it says to which class the phases belong to. 
        bin_class = ph_analysis.assign_bin_class(epochs_bip_fil_phase, bin_num = bin_num)
        phase_bin_means = ph_analysis.get_phase_bin_mean(epochs_bip_fil_phase, bin_class, bin_num = bin_num)
        
        
    
    #%% Extracting ERP amplitude for frequency and phases according to bipolar channel.
        
        # Subj_path is added to exdir, so the EEG epoch files and bipolar  channels are selected from the same subject. 
        exdir_epoch = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
        subj_path =  [d for d in os.listdir(exdir_epoch) if subject_info[-3] in d]
        files_epoch_name = Path(exdir_epoch + subj_path[0])
    
        
        num_epoch = bin_class.T.shape[0]
        if freq_step == '4Hz step':
            freq_band =  list(range(4, 41, 4))
        elif freq_step == '1Hz step':
            freq_band = list(range(4, 41))
            
        
                

            
        labels, ch_names = ph_analysis.get_ERP_1st_2nd(plot = False)
        

        epochs_eeg = mne.read_epochs(files_epoch_name, preload=True).copy().pick_types(eeg=True)
        
        # Choosing the epochs of the bipolar signals accoeding to epochs of EEG channels. So the rejected epochs won't be counted. 
        epochs_num_eeg = epochs_eeg.selection
        epochs_num_bip = epochs_bip.selection
        epochs_num_bip_eeg, epochs_bip_ind,_ = np.intersect1d(epochs_num_bip, epochs_num_eeg, return_indices=True)
        eeg_epoch_filt_phase, target_freq = ph_analysis.extract_phase(epochs_bip._data[epochs_bip_ind , 1, :], 1000, freq_step, 2, angletype = 'degree')
        
        bin_class = ph_analysis.assign_bin_class(eeg_epoch_filt_phase, bin_num = bin_num)
    
        erp_amplitude = {}
        for i_ch, ch in enumerate(labels):
            # Finding erp amplitudes
            labels_eeg_epoch_freq_ph_bin = {}
            erp_amplitude[str(i_ch)] = {}
            
            for i_freq, value_freq in enumerate(freq_band):    
                labels_eeg_epoch_freq_ph_bin[value_freq] = {}
                erp_amplitude[str(i_ch)][value_freq] = {}
                
                for i_bin, value_bin in enumerate(list(range(0,bin_num))):           
                    labels_eeg_epoch_freq_ph_bin[value_freq][value_bin] = np.where(bin_class[i_freq,:] == value_bin)[0]    
                    # 1st mean avg over epochs      
                    erp_amplitude[str(i_ch)][value_freq][value_bin] = np.mean(epochs_eeg._data[labels_eeg_epoch_freq_ph_bin[value_freq][value_bin]][:, :, labels[i_ch]: (labels[i_ch] + 3)], axis = 0)
                    
                    peaks_freq[i_files, freq_step_i, i_ch, i_freq, i_bin] =  np.mean(np.mean(epochs_eeg._data[labels_eeg_epoch_freq_ph_bin[value_freq][value_bin]][:, :, labels[i_ch]: (labels[i_ch] + 3)], axis = 0), axis =1)
 
                    
# Peaks dimension
# [n_files, 1Hz or 4Hz, n_ERPs, n_freqs, n_bins]
# [27, 0 or 1, 2, 10 or 41, 10]
 
    
                
 #%% Plots for the presentation
#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

adjacency_mat,ch_names = mne.channels.find_ch_adjacency(epochs_eeg.info , 'eeg')

thresholds = [3, 3.8]
peaks_tval = np.zeros([64,2])
# choosing bin number 0 and 4 that correspond to 0 and 180
clusters, mask = permutation_cluster(peaks_freq[:, 0, :, :, [0,4],:], adjacency_mat, thresholds = thresholds)
nsubj, _, npeaks,  nfreqs, nphas, nchans, = np.shape(peaks_freq)    
allclusters = np.zeros([nchans, npeaks])
# get the t values for each of the peaks for plotting the topoplots
for p in range(len(clusters)):
    allclusters[:,p] = clusters[p][0]
    
# set all other t values to 0 to focus on clusters
allclusters[mask==False] = 0
cluster_pv = np.zeros([len(clusters)])
for p in range(len(clusters)):
    peaks_tval[:,p] = clusters[p][0]
    if len(clusters[p][2]) >1:
        cluster_pv[p] = min(clusters[p][2])
    else:
        cluster_pv[p] = clusters[p][2]
    
    
fig, sps, cb = plot_topomap_peaks_second_v(peaks_tval, mask, ch_names, [-5,5])
fig.suptitle('All Frequencies and Peak and trough Phases Vs zero', fontsize = 14)
#fig.suptitle('All Frequencies and all phases', fontsize = 14)
sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n  cluster_pv = {cluster_pv[0]}')
sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = {cluster_pv[1]}')
cb.set_label('t-value', rotation = 90)


# positive peak
mean_peaks_sub = np.mean(peaks_freq[:, 0, :, :, 0,:], (0,-2))
fig, sps, cb = plot_topomap_peaks_second_v(mean_peaks_sub.T, np.ones([64,2]), ch_names, [-1,1])
fig.suptitle('All Frequencies ', fontsize = 14)
sps[0].title.set_text(f' \n\n ERP 1')
sps[1].title.set_text(f' \n\n ERP 2')
sps[0].set_ylabel('Positive \n Peak', rotation = 0,  size='x-large',  labelpad=30)
cb.set_label(u"\u03bcv", rotation = 90)



# negative peak
mean_peaks_sub = np.mean(peaks_freq[:, 0, :, :, 4,:], (0,-2))
fig, sps, cb = plot_topomap_peaks_second_v(mean_peaks_sub.T, np.ones([64,2]), ch_names, [-1,1])
fig.suptitle('All Frequencies ', fontsize = 14)
sps[0].title.set_text(f' \n\n ERP 1')
sps[1].title.set_text(f' \n\n ERP 2')
sps[0].set_ylabel('Negative \n Peak', rotation = 0,  size='x-large',  labelpad=30)
cb.set_label(u"\u03bcv", rotation = 90)



# positive - negative peak
mean_peaks_sub = np.mean(peaks_freq[:, 0, :, :, [0,4],:], (1,-2))
fig, sps, cb = plot_topomap_peaks_second_v(((mean_peaks_sub[0,:,:] - mean_peaks_sub[1,:,:])).T, np.ones([64,2]), ch_names, [-1,1])
fig.suptitle('All Frequencies ', fontsize = 14)
sps[0].title.set_text(f' \n\n ERP 1')
sps[1].title.set_text(f' \n\n ERP 2')
sps[0].set_ylabel('Positive \n -\n Negative ', rotation = 0,  size='x-large',  labelpad=30)
cb.set_label(u"\u03bcv", rotation = 90)
               




#%% My cluster


thresholds= [2,2.8]
clusters, mask = permutation_cluster_peak_vs_trough_new(peaks_freq[:, 0, :, :, [0,7],:], adjacency_mat ,freq_band = 'all'  ,thresholds= thresholds )

# check wether it's an empty array


allclusters = np.zeros([nchans, npeaks])
# get the t values for each of the peaks for plotting the topoplots
for p in range(len(clusters)):
    allclusters[:,p] = clusters[p][0]
    
# set all other t values to 0 to focus on clusters
allclusters[mask==False] = 0

cluster_pv = np.zeros([len(clusters)])
for p in range(len(clusters)):
    peaks_tval[:,p] = clusters[p][0]
    if len(clusters[p][2]) >1:
        cluster_pv[p] = min(clusters[p][2])
    elif len(clusters[p][2]) ==1:
        cluster_pv[p] = clusters[p][2]
    else:
        cluster_pv[p] = 0
        
fig, sps, cb = plot_topomap_peaks_second_v(peaks_tval, mask, ch_names, [-5,5])        
fig.suptitle('All Frequencies and Phase Peak Vs trough ', fontsize = 14)

sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n cluster_pv = {cluster_pv[0]}')
sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = {cluster_pv[1]}')
cb.set_label('t-value', rotation = 90)

#%%



thresholds= [2,2.8]

mask = {}
clusters = {}

peaks_tval = np.zeros([64,2])

unique_freqs = np.arange(1, 41, 1)   

for ifreq, freq in enumerate(unique_freqs):

    clusters[str(freq)], mask[str(freq)] = permutation_cluster_peak_vs_trough_new(peaks_freq[:, 0, :, ifreq, [0,3],:], adjacency_mat, freq_band = 'f{freq}', thresholds= thresholds )
     
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters[str(freq)])):
        allclusters[:,p] = clusters[str(freq)][p][0]
        
    # set all other t values to 0 to focus on clusters
    allclusters[mask[str(freq)]==False] = 0
    cluster_pv = np.zeros([len(clusters[str(freq)])])
    for p in range(len(clusters[str(freq)])):
        peaks_tval[:,p] = clusters[str(freq)][p][0]

    
        if len(clusters[str(freq)][p][2]) >1:
            cluster_pv[p] = min(clusters[str(freq)][p][2])
        elif len(clusters[str(freq)][p][2]) ==1:
            cluster_pv[p] = clusters[str(freq)][p][2]
        else:
            cluster_pv[p] = 0

            


            
            
    fig, sps, cb = plot_topomap_peaks_second_v(peaks_tval, mask[str(freq)], ch_names, [-5,5])   
    fig.suptitle(f'{freq}Hz  and Phase Peak Vs trough ', fontsize = 14)
    
    if  cluster_pv[p] < 0.05 :
        sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n cluster_pv = {cluster_pv[0]}')
        sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = {cluster_pv[1]}')
    else :    
        sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n cluster_pv = n.s.')
        sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = n.s')
    
    cb.set_label('t-value', rotation = 90)
#%% Plot Retrospective analysis 1 Hz step
thresholds= [2,2.8]

mask = {}
clusters = {}
cluster_pv_freq =  np.zeros([len(unique_freqs)+1,2])
peaks_tval = np.zeros([64,2])

unique_freqs = np.arange(1, 41, 1)   

for ifreq, freq in enumerate(unique_freqs):

    clusters[str(freq)], mask[str(freq)] = permutation_cluster_peak_vs_trough_new(peaks_freq[:, 0, :, ifreq, [0,3],:], adjacency_mat, freq_band = 'f{freq}', thresholds= thresholds )
     
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters[str(freq)])):
        allclusters[:,p] = clusters[str(freq)][p][0]
        
    # set all other t values to 0 to focus on clusters
    allclusters[mask[str(freq)]==False] = 0
    cluster_pv = np.zeros([len(clusters[str(freq)])])
    for p in range(len(clusters[str(freq)])):
        peaks_tval[:,p] = clusters[str(freq)][p][0]

    
        if len(clusters[str(freq)][p][2]) >1:
            cluster_pv[p] = min(clusters[str(freq)][p][2])
        elif len(clusters[str(freq)][p][2]) ==1:
            cluster_pv[p] = clusters[str(freq)][p][2]
        else:
            cluster_pv[p] = 0

            
        if  cluster_pv[p] < 0.05 :
            cluster_pv_freq[freq, p] = cluster_pv[p]


# Peak is 0 or 1, since we have 2 ERPs

plot_retro_peak_vs_trough_1hz(clusters, ch_names, mask, 0, cluster_pv_freq)
plot_retro_peak_vs_trough_1hz(clusters, ch_names, mask, 1, cluster_pv_freq)


#%% Plot Retrospective analysis 4 Hz step
thresholds= [2,2.8]

mask = {}
clusters = {}
cluster_pv_freq =  np.zeros([len(unique_freqs)+1,2])
peaks_tval = np.zeros([64,2])

unique_freqs = np.arange(1, 41, 1)   

for ifreq, freq in enumerate(unique_freqs):

    clusters[str(freq)], mask[str(freq)] = permutation_cluster_peak_vs_trough_new(peaks_freq[:, 1, :, ifreq, [0,4],:], adjacency_mat, freq_band = 'f{freq}', thresholds= thresholds )
     
    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters[str(freq)])):
        allclusters[:,p] = clusters[str(freq)][p][0]
        
    # set all other t values to 0 to focus on clusters
    allclusters[mask[str(freq)]==False] = 0
    cluster_pv = np.zeros([len(clusters[str(freq)])])
    for p in range(len(clusters[str(freq)])):
        peaks_tval[:,p] = clusters[str(freq)][p][0]

    
        if len(clusters[str(freq)][p][2]) >1:
            cluster_pv[p] = min(clusters[str(freq)][p][2])
        elif len(clusters[str(freq)][p][2]) ==1:
            cluster_pv[p] = clusters[str(freq)][p][2]
        else:
            cluster_pv[p] = 0

            
        if  cluster_pv[p] < 0.05 :
            cluster_pv_freq[freq, p] = cluster_pv[p]


# Peak is 0 or 1, since we have 2 ERPs

def plot_retro_peak_vs_trough_4hz(clusters, ch_names, mask,  cluster_pv_freq):

    maskparam=None
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20,8))
    fig.suptitle('Retrospective Analysis, 4 Hz Step', fontsize = 18, fontweight="bold")
    cols = [format(col) for col in np.arange(1,11,1)]
    rows = ['{}'.format(row) for row in [ 'ERP1\n\n','ERP2\n\n']]
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(f'{np.multiply(int(col), 4)} Hz', size =18, fontweight="bold")   
        im = topoplot_2d (ch_names, clusters[str(col)][0][0], 
                                     clim=[-5,5], axes=ax, 
                                     mask=mask[str(col)][:,0], maskparam=maskparam)
        fig.subplots_adjust(wspace=0, hspace=0)
        cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    
    
    for ax, col in zip(axes[1], cols):
        ax.set_title(f'{np.multiply(int(col), 4)} Hz', size=18, fontweight="bold")   
        im = topoplot_2d (ch_names, clusters[str(col)][1][0], 
                                     clim=[-5,5], axes=ax, 
                                     mask=mask[str(col)][:,1], maskparam=maskparam)
        fig.subplots_adjust(wspace=0, hspace=0)
        cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    
    
    
    
    for i_l, l in enumerate(np.where(cluster_pv_freq[:,0]>0)[0]):
    
        if l < 10:
            row = 0; col =l
        axes[row, (col-1)].set_xlabel(f' P = {cluster_pv_freq[l, 0]}')
        
        
    for i_l, l in enumerate(np.where(cluster_pv_freq[:,1]>0)[0]):
    
        if l < 10:
            row = 1; col =l
        axes[row, (col-1)].set_xlabel(f' P = {cluster_pv_freq[l, 1]}')    


    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size=18, fontweight="bold")
        
    plt.tight_layout()
    
    
    
plot_retro_peak_vs_trough_4hz(clusters, ch_names, mask, cluster_pv_freq)

    