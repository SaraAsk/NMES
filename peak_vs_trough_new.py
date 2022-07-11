#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:41:37 2022

@author: sara
"""










def permutation_cluster_peak_vs_trough_new(peaks, adjacency_mat, thresholds, freq_band):
        # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    
    if freq_band == 'all':
    
        mean_peaks_freq = np.mean(peaks, ( -1))
        
    else:
        
        mean_peaks_freq = peaks
    
    #first row : peak - trough
    # According to mne "The first dimension should correspond to the difference between paired samples (observations) in two conditions. "
    mean_peaks_phase = mean_peaks_freq[:, :, :, 0] -  mean_peaks_freq[:, :, :, 1]
    
    mean_peaks_phase = np.mean(mean_peaks_freq, ( -1))
    # get matrix dimensions
    nsubj, nchans, npeaks, = np.shape(mean_peaks_phase)
    nperm = 100
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([nperm+1, npeaks])
    thresholds=thresholds
    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster

   
    for p in range(npeaks):
        mean_peaks_phase = mean_peaks_freq[:, :, p, 0] -  mean_peaks_freq[:, :, p, 1]
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










def permutation_cluster(peaks, adjacency_mat, thresholds):
        # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2, -1))
    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([npeaks])
    thresholds=thresholds
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
            # save the original cluster information (1st iteration) 


                # get the channels which are in the main cluster
            mask[:,p] = cluster[1][np.argmax(t_sum)]
        else:
            max_cluster_size[p] = 0
            
            
        clusters.append(cluster)   


    
    return clusters, mask

def permutation_cluster_peak_and_trough_against_zero(peaks, adjacency_mat, thresholds):
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-1))
    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([npeaks])
    thresholds=thresholds
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


def Select_Epochs_peak_trough(epochs, freq, phase):
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



#%%

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import mne.stats
import mne


# phase analysis function
import phase_analysis_function as ph_analysis



exdir = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
files = Path(exdir).glob('*.fif*')
save_folder_fig = "/home/sara/NMES/analyzed data/phase_analysis/Figures/peak_vs_trough/"

plt.close('all')
#idx =263


labels, _ = ph_analysis.get_ERP_1st_2nd(plot = True)

# num_sub, channels, peaks, phases(0, 180), all freq

peaks = np.zeros([27, 64, 2, 8, 10])
#peaks = np.zeros([27, 64, 2, 8, 10]) # for all phases 

#unique_phases_cosine = np.arange(0, 360, 45 ) 
unique_phases_cosine = np.arange(0, 360, 45 ) #put a flag here
#unique_phases_sine = np.arange(90, 450, 180 )
unique_freqs = np.arange(4, 44, 4)    
peaks_tval = np.zeros([64,2])
for ifiles, f in enumerate(files):
    epochs = mne.read_epochs(f, preload=True).copy().pick_types( eeg=True)
    epochs_byfreqandphase = {} 
    erp_byfreqandphase = {} 
    peaks_byfreqandphase = {}

    
    for ifreq, freq in enumerate(unique_freqs):
        print(ifreq, freq)
        erp_byfreqandphase[str(freq)]  = {}
        epochs_byfreqandphase[str(freq)] = {}
        peaks_byfreqandphase[str(freq)] = {} 
        for iphase, phase in enumerate(unique_phases_cosine):
            sel_idx = Select_Epochs_peak_trough(epochs, freq, phase)
            epochs_byfreqandphase[str(freq)][str(phase)] = epochs[sel_idx]
            erp_byfreqandphase[str(freq)][str(phase)]  = epochs_byfreqandphase[str(freq)][str(phase)].average() 
            #peaks_byfreqandphase[str(freq)][str(phase)] = np.mean((erp_byfreqandphase[str(freq)][str(phase)]._data[:,idx:(idx+3)]),1)             
            for ipeak, peak in enumerate(labels):

                peaks_byfreqandphase[str(freq)][str(phase)] = np.mean((erp_byfreqandphase[str(freq)][str(phase)]._data[:,peak:(peak + 3)]),1)
                # To remove none arrays after selecting epochs
                if str(erp_byfreqandphase[str(freq)][str(phase)].comment) == str(''):
                    peaks_byfreqandphase[str(freq)][str(phase)] = np.zeros(64) 
                else:
                    peaks[ifiles, :, ipeak, iphase, ifreq] = peaks_byfreqandphase[str(freq)][str(phase)] 

mask = {}
pvals = {}
clusters = {}

adjacency_mat,_ = mne.channels.find_ch_adjacency(epochs_byfreqandphase[str(freq)][str(phase)].info , 'eeg')



#%% Plots for the presentation
#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thresholds = [3, 3.8]
clusters, mask = permutation_cluster(peaks, adjacency_mat, thresholds = thresholds)
nsubj, nchans, npeaks, nphas, nfreqs = np.shape(peaks)    
allclusters = np.zeros([nchans, npeaks])
# get the t values for each of the peaks for plotting the topoplots
for p in range(len(clusters)):
    allclusters[:,p] = clusters[p][0]
    
# set all other t values to 0 to focus on clusters
allclusters[mask==False] = 0
ch_names = epochs.ch_names
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
mean_peaks_sub = np.mean(peaks, (0,-1))
fig, sps, cb = plot_topomap_peaks_second_v(mean_peaks_sub[:,:,0], np.ones([64,2]), ch_names, [-1,1])
fig.suptitle('All Frequencies ', fontsize = 14)
sps[0].title.set_text(f' \n\n ERP 1')
sps[1].title.set_text(f' \n\n ERP 2')
sps[0].set_ylabel('Positive \n Peak', rotation = 0,  size='x-large',  labelpad=30)
cb.set_label(u"\u03bcv", rotation = 90)



# negative peak
mean_peaks_sub = np.mean(peaks, (0,-1))
fig, sps, cb = plot_topomap_peaks_second_v(mean_peaks_sub[:,:,1], np.ones([64,2]), ch_names, [-1,1])
fig.suptitle('All Frequencies ', fontsize = 14)
sps[0].title.set_text(f' \n\n ERP 1')
sps[1].title.set_text(f' \n\n ERP 2')
sps[0].set_ylabel('Negative \n Peak', rotation = 0,  size='x-large',  labelpad=30)
cb.set_label(u"\u03bcv", rotation = 90)



# positive - negative peak
mean_peaks_sub = np.mean(peaks, (0,-1))
fig, sps, cb = plot_topomap_peaks_second_v((mean_peaks_sub[:,:,0] - mean_peaks_sub[:,:,1]), np.ones([64,2]), ch_names, [-1,1])
fig.suptitle('All Frequencies ', fontsize = 14)
sps[0].title.set_text(f' \n\n ERP 1')
sps[1].title.set_text(f' \n\n ERP 2')
sps[0].set_ylabel('Positive \n -\n Negative ', rotation = 0,  size='x-large',  labelpad=30)
cb.set_label(u"\u03bcv", rotation = 90)




#%% My cluster


thresholds= [2,2.8]
clusters, mask = permutation_cluster_peak_vs_trough_new(peaks[:,:, :, [0,3],: ], adjacency_mat ,freq_band = 'all'  ,thresholds= thresholds )

# check wether it's an empty array


allclusters = np.zeros([nchans, npeaks])
# get the t values for each of the peaks for plotting the topoplots
for p in range(len(clusters)):
    allclusters[:,p] = clusters[p][0]
    
# set all other t values to 0 to focus on clusters
allclusters[mask==False] = 0
ch_names = epochs.ch_names
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












#%% Peak Vs Trough for each freq

thresholds= [2,2.8]

mask = {}
clusters = {}
cluster_pv_freq =  np.zeros([len(unique_freqs),2])
for ifreq, freq in enumerate(unique_freqs):

    clusters[str(ifreq)], mask[str(ifreq)] = permutation_cluster_peak_vs_trough_new(peaks[:, :, :, [6,7], ifreq], adjacency_mat, freq_band = 'f{freq}', thresholds= thresholds )

    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters[str(ifreq)])):
        allclusters[:,p] = clusters[str(ifreq)][p][0]
        
    # set all other t values to 0 to focus on clusters
    allclusters[mask[str(ifreq)]==False] = 0
    ch_names = epochs.ch_names
    cluster_pv = np.zeros([len(clusters[str(ifreq)])])
    for p in range(len(clusters[str(ifreq)])):
        peaks_tval[:,p] = clusters[str(ifreq)][p][0]
        if len(clusters[str(ifreq)][p][2]) >1:
            cluster_pv[p] = min(clusters[str(ifreq)][p][2])
        elif len(clusters[str(ifreq)][p][2]) ==1:
            cluster_pv[p] = clusters[str(ifreq)][p][2]
        else:
            cluster_pv[p] = 0
            
            
        if  cluster_pv[p] < 0.05 :
            cluster_pv_freq[ifreq, p] = cluster_pv[p]


            
            


def plot_retro_peak_vs_trough_4hz(clusters, ch_names, mask,  cluster_pv_freq):
    maskparam=None
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20,8))
    fig.suptitle('Real-time Analysis peak vs trough', fontsize = 18, fontweight="bold")
    cols = [format(col) for col in np.arange(0,10,1)]
    rows = ['{}'.format(row) for row in [ 'ERP1\n\n','ERP2\n\n']]
    
    for ax, col in zip(axes[0], cols):
        ax.set_title(f'{np.multiply(int(col) +1, 4)} Hz', size =18, fontweight="bold")   
        im = topoplot_2d (ch_names, clusters[str(col)][0][0], 
                                     clim=[-5,5], axes=ax, 
                                     mask=mask[str(col)][:,0], maskparam=maskparam)
        fig.subplots_adjust(wspace=0, hspace=0)
        cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    
    
    for ax, col in zip(axes[1], cols):
        ax.set_title(f'{np.multiply(int(col) +1, 4)} Hz', size=18, fontweight="bold")   
        im = topoplot_2d (ch_names, clusters[str(col)][1][0], 
                                     clim=[-5,5], axes=ax, 
                                     mask=mask[str(col)][:,1], maskparam=maskparam)
        fig.subplots_adjust(wspace=0, hspace=0)
        cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
        cb.ax.tick_params(labelsize=12)
    
    
    
    
    for i_l, l in enumerate(np.where(cluster_pv_freq[:,0]>0)[0]):
    
        if l < 10:
            row = 0; col =l
        axes[row, (col)].set_xlabel(f' P = {cluster_pv_freq[(col), 0]}')
        
        
    for i_l, l in enumerate(np.where(cluster_pv_freq[:,1]>0)[0]):
    
        if l < 10:
            row = 1; col =l
        axes[row, (col)].set_xlabel(f' P = {cluster_pv_freq[(col), 1]}')    


    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=0, size=18, fontweight="bold")
        
    plt.tight_layout()
    
    
    
plot_retro_peak_vs_trough_4hz(clusters, ch_names, mask, cluster_pv_freq)


#%% Cluster permuattion test for peak and trough against zero



thresholds= [2,2.8]

mask = {}
clusters = {}

for ifreq, freq in enumerate(unique_freqs):

    clusters[str(freq)], mask[str(freq)] = permutation_cluster_peak_and_trough_against_zero(peaks[:, :, :, :, ifreq], adjacency_mat,  thresholds= thresholds )

    allclusters = np.zeros([nchans, npeaks])
    # get the t values for each of the peaks for plotting the topoplots
    for p in range(len(clusters[str(freq)])):
        allclusters[:,p] = clusters[str(freq)][p][0]
        
    # set all other t values to 0 to focus on clusters
    allclusters[mask[str(freq)]==False] = 0
    ch_names = epochs.ch_names
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
    fig.suptitle(f'{freq}Hz  and Phase Peak and trough against zero ', fontsize = 14)
    

    sps[0].title.set_text(f' \n\n ERP 1\n\n TH = {thresholds[0]} \n\n cluster_pv = {cluster_pv[0]}')
    sps[1].title.set_text(f' \n\n ERP 2\n\n TH = {thresholds[1]} \n\n  cluster_pv = {cluster_pv[1]}')

    cb.set_label('t-value', rotation = 90)
    fig.savefig(save_folder_fig +'cosine' + '/each_freq/' +'each_freq' + '_' + str(freq) + '_' + 'Cosine' + '.png')




#%%







import pickle

with open('/home/sara/NMES/analyzed data/phase_analysis/pickle/cosine_fit_all_subjects.p', 'rb') as f:
    x = pickle.load(f)
    
    
# We have 21 subjects in totals, but for some of them these xdf files is being recorded seperately.   

freq_steps = ['1Hz step', '4Hz step'] 
data = {}
surrogate = {}
phi = {}
mag = {}

for num_sub in range(len(x)):
    data[str(num_sub)] = {}
    surrogate[str(num_sub)] = {}
    phi[str(num_sub)] = {}
    mag[str(num_sub)] = {}
    
    for freq_step_i, freq_step_hz in enumerate(freq_steps): 
        data[str(num_sub)][str(freq_step_i)] = {}
        surrogate[str(num_sub)][str(freq_step_i)] = {}
        phi[str(num_sub)][str(freq_step_i)] = {}
        mag[str(num_sub)][str(freq_step_i)] = {}

        if freq_step_hz == '4Hz step':
            freq_band =  list(range(4, 41, 4))
        elif freq_step_hz == '1Hz step':
            freq_band = list(range(4, 41))

        
        for i in range(2): # len erp amplitude
            data[str(num_sub)][str(freq_step_i)][str(i)] = {}
            surrogate[str(num_sub)][str(freq_step_i)][str(i)]  = {}
            phi[str(num_sub)][str(freq_step_i)][str(i)]  = {}
            mag[str(num_sub)][str(freq_step_i)][str(i)]  = {}

            for jf, freq in enumerate(freq_band):  
                data[str(num_sub)][str(freq_step_i)][str(i)][str(freq)] = x[num_sub][freq_step_i][str(i)][str(freq)][0]['data']

                np.searchsorted(np.array(list(x[num_sub][freq_step_i][str(i)][str(freq)][0]['data'].keys())), 0)
    






#%% Can I get the peaks for peak and trough phases and frequencies from X pickle file?



with open('/home/sara/NMES/analyzed data/phase_analysis/pickle/cosine_fit_all_subjects.p', 'rb') as f:
    x = pickle.load(f)
    
unique_phases_cosine = np.arange(0, 360, 180 ) #put a flag here   
# We have 21 subjects in totals, but for some of them these xdf files is being recorded seperately.   
freq_steps = ['1Hz step', '4Hz step'] 
freq_steps = ['1Hz step'] 
erp_amp = {}

for num_sub in range(len(x)):
    erp_amp[str(num_sub)] = {}
    
    
    for freq_step_i, freq_step_hz in enumerate(freq_steps): 
        erp_amp[str(num_sub)][str(freq_step_hz)] = {}

        if freq_step_hz == '4Hz step':
            freq_band =  list(range(4, 41, 4))
        elif freq_step_hz == '1Hz step':
            freq_band = list(range(4, 41))

        
        for i in range(2): # len erp amplitude
            erp_amp[str(num_sub)][str(freq_step_hz)][str(i)] = {}

            for jf, freq in enumerate(freq_band):  
                erp_amp[str(num_sub)][str(freq_step_hz)][str(i)][str(freq)] = {}
                
                for iphase, phase in enumerate(unique_phases_cosine):  
                    erp_amp[str(num_sub)][str(freq_step_hz)][str(i)][str(freq)][str(phase)] = {}
                    
                    # phase keys are saved by their original phase value that was calculated
                    phase_keys = [x_phase+360 if x_phase<0 else x_phase for x_phase in np.array(list(x[num_sub][freq_step_i][str(i)][str(freq)][0]['data'].keys()))]
                    phase_peak_idx = (np.abs(np.array(phase_keys) - phase)).argmin()
                    phase_peak_val = np.array(list(x[num_sub][freq_step_i][str(i)][str(freq)][0]['data'].values()))[phase_peak_idx]
                    erp_amp[str(num_sub)][str(freq_step_hz)][str(i)][str(freq)][str(phase)] = phase_peak_val
                


#%%


# Finding the keys of the peak and trough phases and adding 360 if they are negative
phase_keys = [x_phase+360 if x_phase<0 else x_phase for x_phase in np.array(list(x[num_sub][freq_step_i][str(i)][str(freq)][0]['data'].keys()))]
phase_peak_idx = (np.abs(np.array(phase_keys) - 0)).argmin()
phase_trough_idx = (np.abs(np.array(phase_keys) - 180)).argmin()
    
phase_peak_val = np.array(list(x[num_sub][freq_step_i][str(i)][str(freq)][0]['data'].values()))[phase_peak_idx]
phase_trough_val = np.array(list(x[num_sub][freq_step_i][str(i)][str(freq)][0]['data'].values()))[phase_trough_idx]















