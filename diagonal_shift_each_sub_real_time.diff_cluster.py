#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:57:48 2022

@author: sara
"""

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
    
    
    all_channels_clustered, ERP_1_ch, ERP_2_ch = clustering_channels()
       
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





def clustering_channels():
    
    
    exdir = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
    files = Path(exdir).glob('*.fif*')
    plt.close('all')
    #idx =263
    
    
    labels,_ = get_ERP_1st_2nd(plot = True)
    
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
                sel_idx = Select_Epochs(epochs, freq, phase)
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
    clusters, mask, mask_dict = permutation_cluster(peaks, adjacency_mat)
    
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
    plot_topomap_peaks_second_v(a, mask, ch_names, [-5,5])
    
    
    
    # Name and indices of the EEG electrodes that are in the biggest cluster
    all_ch_names_biggets_cluster =  []
    all_ch_ind_biggets_cluster =  []
    
    for p in range(len(clusters)):
        # indices
        all_ch_ind_biggets_cluster.append(np.where(mask[:,p] == 1))
        # channel names
        all_ch_names_biggets_cluster.append([ch_names[i] for i in np.where(mask[:,p] == 1)[0]])
    
    
    all_channels_clustered = all_ch_names_biggets_cluster[0] + all_ch_names_biggets_cluster[1]
    
    
    
    # dict form
    allclusters_dict = {}
    #allmask = {}
    for p in range(len(clusters)):
        allcluster = np.zeros((64, len(clusters[p][1]) ))
        
        for n_clu in range(len(clusters[p][1])):
            allcluster[:, n_clu] = clusters[p][0]
            allcluster[:, n_clu][mask_dict[str(p)][str(n_clu)]==False] = 0
            #allmask[str(p)] = allcluster[:, n_clu][np.where(allcluster[:, n_clu] != 0)]
        allclusters_dict[str(p)] = allcluster
    

        for n_clu in range(len(clusters[p][1])):
            masks = np.zeros([64,1])
            #masks = allclusters_dict[str(p)][:, n_clu][np.where(allclusters_dict[str(p)][:, n_clu] !=0) ]
            masks = mask_dict[str(p)][str(n_clu)] *1
            plot_topomap_peaks(clusters[p][0], masks, ch_names, [-5,5])
            
    all_ch_names_biggets_cluster =  []
    all_ch_ind_biggets_cluster =  []
    
    for p in range(len(clusters)):

        # channels here do not belong to the maximum cluster. Just wanted to try if choosing another cluster will lead to better p values or not
        # based on a different cluster that we get for peak vs trough
        # indices
        all_ch_ind_biggets_cluster.append(np.where(mask_dict[str(p)][str(1)]*1))
        # channel names
        all_ch_names_biggets_cluster.append([ch_names[i] for i in np.where(mask_dict[str(p)][str(1)]*1)[0]])    
        
    all_channels_clustered = all_ch_names_biggets_cluster[0] + all_ch_names_biggets_cluster[1]    
       
    return all_channels_clustered, all_ch_names_biggets_cluster[0], all_ch_names_biggets_cluster[1]





def permutation_cluster(peaks, adjacency_mat):
        # in this function, peaks is a 5 dim matrix with dims, nsubj, nchans, npeaks, nphas, nfreq
    import mne.stats
    # reduce dimensions by averaging over target frequencies and phases
    mean_peaks = np.mean(peaks, (-2, -1))
    # get matrix dimensions
    nsubj, nchans, npeaks = np.shape(mean_peaks)
    nperm = 100
    clusters = []
    mask = np.zeros([nchans, npeaks])
    max_cluster_size = np.zeros([nperm+1, npeaks])
    thresholds=[3.2,3]
    mask_dict = {}
    # get the original cluster size during the first loop
    # perform 1000 random permutations (sign flipping) and each time determine the size of the biggest cluster



 
    for p in range(npeaks):
         mask_dict[str(p)] = {}
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
         
         for t, t_val in enumerate(t_sum):
             mask_dict[str(p)][str(t)] = {}
             print(t, t_val)
             mask_dict[str(p)][str(t)] = cluster[1][t]
             
             
    return clusters, mask, mask_dict



def plot_topomap_peaks_second_v(peaks, mask, ch_names, clim):

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

    plt.show()

    return im








def plot_topomap_peaks(peaks, mask, ch_names, clim):

    import matplotlib.pyplot as plt

    nplots =1 

    maskparam = dict(marker='.', markerfacecolor='k', markeredgecolor='k',
                linewidth=0, markersize=5)

    fig, sps = plt.subplots()
    
    


    im = topoplot_2d (  ch_names, peaks, 
                        clim=clim, 
                        mask=mask, maskparam=maskparam)

    fig.subplots_adjust(wspace=0, hspace=0)
    cb = plt.colorbar(im[0],  ax = sps, fraction=0.02, pad=0.04)
    
    cb.ax.tick_params(labelsize=12)

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
    n1_timewin = [.049, .1]
    n2_timewin= [.108, .246]
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

def do_cosine_fit_ll(erp_amplitude, phase_bin_means, freq_band, labels, perm = True):

    
    cosinefit = {}
    amplitudes_cosine = np.zeros([len(freq_band), len(labels)])
    pvalues_cosine = np.zeros([len(freq_band), len(labels)])

    
    
    #x = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    x = phase_bin_means
    
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
            if  freq_step_i =='Real-time':
                data_erp[str(freq)] = erp_amplitude[str(erp)][str(freq)]   
            else:
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
        
        if erp==0 and str(subject_info[-2] == 'Experiment'): 
             plt.title(f'1st ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder +  'fig_2c_biggest_cluster' + '_' + str(subject_info[-3]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
        elif erp==1 and str(subject_info[-2] == 'Experiment'):  
             plt.title(f'2nd ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_biggest_cluster' + '_' + str(subject_info[-3]) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
         
            
         

        fig = plt.figure()
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
        
        if erp==0 and str(subject_info[-2] == 'Experiment' ):
             plt.title(f'1st ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
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

        if i==0:
             plt.title(f'1st ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
        else: 
             plt.title(f'2nd ERP, Subject: {subject_info[-3], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-3]) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
    
    return fig








def plot_torrecillos_2c(mod_depth, surrogate, i, x):
    from textwrap import wrap
    from pandas.plotting import table
    from mne.stats import permutation_cluster_test
   

    amp_erp_all = [] 
    surrogate_erp_all = []  
    for num_sub in range(len(x)):   
        amp_erp_all.append(np.array(list(mod_depth[str(num_sub)][str(i)].values())))
        surrogate_erp_all.append(np.mean(np.array(list(surrogate[str(num_sub)][str(i)].values())), axis =1))
        
    amp_erp_all_arr = np.array(amp_erp_all)
    surrogate_erp_all_arr = np.array(surrogate_erp_all)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(freq_band, np.mean(amp_erp_all_arr, axis = 0 ), color = 'k', label='Real data')
    ax.plot(freq_band, np.mean(surrogate_erp_all_arr, axis = 0 ), color = 'r', label='Surrogate')
    
    
    ax.fill_between(freq_band, np.std(amp_erp_all_arr, axis = 0 ) + np.mean(amp_erp_all_arr, axis = 0 ),\
                      np.mean(amp_erp_all_arr, axis = 0 ) - np.std(amp_erp_all_arr, axis = 0 ),\
                          color = '0.8')
    
    ax.fill_between(freq_band, np.std(surrogate_erp_all_arr, axis = 0 ) + np.mean(surrogate_erp_all_arr, axis = 0 ),\
                      np.mean(surrogate_erp_all_arr, axis = 0 ) - np.std(surrogate_erp_all_arr, axis = 0 ),\
                           alpha = 0.18, color = 'r')
    if i==0 :
        erp_info= 'ERP1, Real-Time'
    elif i==1 :
        erp_info= 'ERP2, Real-Time' 

        
    plt.title("\n".join(wrap((f'Per subject: Strength of Modulation {erp_info}'), 38)))
    ax.set_ylabel('Strength of Mod.') 
    ax.set_xlabel('Frequencies (Hz)')    
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=-0.1, top=1.7)

    
    threshold = 4
    

    
    
    # cluster permutation test
    T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([amp_erp_all_arr, surrogate_erp_all_arr], n_permutations=1000,
                             threshold=threshold, tail=1, n_jobs=1,
                             out_type='mask')

    for i_c, c in enumerate(clusters):
        print(i_c, c)
     
        c = c[0]
        if cluster_p_values[i_c] <= 0.06 and (freq_band[c.stop - 1] - freq_band[c.start]) > 0:
            h = plt.axvspan(freq_band[c.start], freq_band[c.stop - 1], ymin = 0.2, ymax = 0.7, 
                            color='g', alpha=0.25)
            print(cluster_p_values[i_c], freq_band[c.start], freq_band[c.stop - 1])
            ax.text(freq_band[c.start], 1.4, f'P-value = {cluster_p_values[i_c]}', color='k') 

            
        else:
            plt.axvspan(freq_band[c.start], freq_band[c.stop - 1], color=(0.3, 0.3, 0.3),
                        alpha=0.3)
    
# =============================================================================
#     if i==1:
#         df2 = pd.DataFrame(np.array([[freq_band[clusters[0][0].start], freq_band[clusters[0][0].stop-1]], [cluster_p_values[0], cluster_p_values[1]]]), columns=['Cluster start', 'Cluster end'],  index=['Star-End freq', 'P value'])
#         
#         df2 = pd.DataFrame(np.array([[freq_band[clusters[0][0].start], freq_band[clusters[0][0].stop-1]], [cluster_p_values[0], cluster_p_values[1]]]), columns=['Cluster start', 'Cluster end'],  index=['Star-End freq', 'P value'])
# 
#         table6  =table(ax, df2, loc="upper right");
#         table6.set_fontsize(40)
#         table6.scale(0.5,0.5)
# =============================================================================
        
    
        
    return fig    









#%%

import mne
import math
import lmfit
import itertools  
import mne.stats
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.stats import sem # standard error of the mean.
from scipy.stats import zscore
import matplotlib.pyplot as plt
from multiprocessing import Pool
import seaborn as sns; sns.set_theme()
from scipy.interpolate import interp1d

import phase_analysis_function as ph_analysis




exdir_epoch_GA = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
files_GA = list(Path(exdir_epoch_GA).glob('*.fif*'))
save_folder_fig = "/home/sara/NMES/analyzed data/phase_analysis/Figures/cluster_freq/each_subject"
save_folder_pickle =  "/home/sara/NMES/analyzed data/phase_analysis/pickle/"



amplitudes_cosines_all_subjects = []
amplitudes_cosines_all_subjects_LL = []
all_subjects_names = []

cosine_fit_all_subjects = []
cosine_fit_all_subjects_LL = []

# Group avg-real-time
epochs_eeg, all_channels_clustered, ERP1_chan, ERP2_chan = epoch_concat_clustered_and_mod_dict(files_GA)





    
    
labels, ch_names = get_ERP_1st_2nd(plot = False)
    


    
# Getting the indices of fisrt and second ERP in epochs_eeg
_, _, ERP1_ch_indx = np.intersect1d( ERP1_chan, epochs_eeg.info['ch_names'], return_indices=True  )
_, _, ERP2_ch_indx = np.intersect1d( ERP2_chan, epochs_eeg.info['ch_names'], return_indices=True  )

ERP1_ch_indx =  np.sort(ERP1_ch_indx)
ERP2_ch_indx =  np.sort(ERP2_ch_indx)
ERP_indexs = [ERP1_ch_indx.T, ERP2_ch_indx.T]

        
  
#%%  

for f in files_GA:
    subject_info = f.parts 

    
        
    # Subj_path is added to exdir, so the EEG epoch files and bipolar  channels are selected from the same subject. 


    epochs_eeg = mne.read_epochs(f, preload=True).copy().pick_types(eeg=True)
        

        
                

    
    
    # 4 Hz step with lucky loop labels
    epochs_byfreqandphase = {} 
    erp_amplitude_ll = {}
    ERP_byfreqandphase = {}
    evoked_zscored = {}
    
    for i_ch, ch in enumerate(ERP_indexs):
        epochs_byfreqandphase[str(i_ch)] = {} 
        erp_amplitude_ll[str(i_ch)] = {}
        ERP_byfreqandphase[str(i_ch)] = {}
        evoked_zscored[str(i_ch)] = {}
        for freq in np.arange(4,44,4):
            epochs_byfreqandphase[str(i_ch)][str(freq)] = {}
            ERP_byfreqandphase[str(i_ch)][str(freq)] = {}
            evoked_zscored[str(i_ch)][str(freq)] = {}
            for phase in np.arange(0,360,45):
                sel_idx = Select_Epochs(epochs_eeg, freq, phase)
                epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = epochs_eeg[sel_idx]
                ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)]._data[:, ch, labels[i_ch]]
                evoked_zscored[str(i_ch)][str(freq)][str(phase)] = np.mean(np.mean(ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)], axis = 1))
                if str(evoked_zscored[str(i_ch)][str(freq)][str(phase)]) == 'nan':
                    evoked_zscored[str(i_ch)][str(freq)][str(phase)] = 0
            
    cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll = do_cosine_fit_ll(evoked_zscored, np.arange(0,360,45), np.arange(4,44,4), labels, perm = True)
    
    amplitudes_cosines_all_subjects_LL.append(amplitudes_cosine_ll)
    cosine_fit_all_subjects_LL.append(cosinefit_ll)
    
    if not (cosinefit_ll[str(0)] ):
        print(f'There are not enough epochs by freq and phase for Subject: {subject_info[-3]}')
    else:
        
        fig_2c_ll = fig_2c_plot(evoked_zscored, np.arange(4,44,4), cosinefit_ll, subject_info, 'Real-time', save_folder_fig)
        fig_2a_ll = fig_2a_plot(evoked_zscored, np.arange(4,44,4),  subject_info, 'Real-time', save_folder_fig,  vmin = -2, vmax= 2)
 
            

# Writing and then loading the pickle files again    
cosine_fit_ll = 'cosine_fit_all_subjects_LL_changed_cluster' + '.p'
with open(str(save_folder_pickle) + cosine_fit_ll, 'wb') as fp:
    pickle.dump(cosine_fit_all_subjects_LL, fp, protocol=pickle.HIGHEST_PROTOCOL)        
    
      
    

with open('/home/sara/NMES/analyzed data/phase_analysis/pickle/cosine_fit_all_subjects_ll.p', 'rb') as f:
    x = pickle.load(f)
    

        
         

#%% Single subject
    

x = cosine_fit_all_subjects_LL

  
# We have 21 subjects in totals, but for some of them these xdf files is being recorded seperately.   

freq_band = np.arange(4, 44, 4)
mod_depth = {}
surrogate = {}
phi = {}
mag = {}

for num_sub in range(len(x)):
    print(num_sub)
    mod_depth[str(num_sub)] = {}
    surrogate[str(num_sub)] = {}
    phi[str(num_sub)] = {}
    mag[str(num_sub)] = {}
    

        
    for i in range(2): # len erp amplitude
        mod_depth[str(num_sub)][str(i)] = {}
        surrogate[str(num_sub)][str(i)]  = {}
        phi[str(num_sub)][str(i)]  = {}
        mag[str(num_sub)][str(i)]  = {}

        for jf, freq in enumerate(freq_band):  
            mod_depth[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['amp']
            surrogate[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['surrogate']
            phi[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['Fit'].best_values['phi']
            mag[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['Fit'].best_fit
            



# i= 0 ; freq_step_i = 0 # ERP1, 1Hz step

plot_torrecillos_2c(mod_depth, surrogate, 0, x) # ERP1
plot_torrecillos_2c(mod_depth, surrogate, 1, x) # ERP2


        
#%% Optimal phase distributation

 
# I just plot it for 4 hz step
mag_erp = []
for i in range(len(ERP_indexs)):

    mag_all = [] 
    for num_sub in range(len(x)):   
        mag_all.append(np.array(list(mag[str(num_sub)][str(i)].values())))

    mag_erp.append(np.mean(mag_all, axis =0))



titles = ['Freq 4', 'Freq 8', 'Freq 12', 'Freq 16', 'Freq 20', 'Freq 24', 'Freq 28', 'Freq 32', 'Freq 36', 'Freq 40'] 

fig = plt.figure(constrained_layout=True)
fig.suptitle('Optimal Phase distribution', fontweight="bold")

# create 3x1 subfigs
subfigs = fig.subfigures(nrows=2, ncols=1)
for row, subfig in enumerate(subfigs):

    subfig.suptitle(f'ERP {row+1}')

    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=10,subplot_kw=dict(projection='polar'))
    for col, ax in enumerate(axs):
        ph_analysis.phase_optimal_distribution(ax, mag_erp[row][col], titles[col])    
 




#%%% Circular correlation
import pycircstat
from scipy import stats








amp_erp_all = np.zeros([len(freq_band), len(x)])
phase_erp_all = np.zeros([len(freq_band), len(x)])

fig, ax =  plt.subplots(1,2)

amp = {}
phase = {}
for i in range(len(ERP_indexs)):
    amp[str(i)] = {}
    phase[str(i)] = {}

    for num_sub in range(len(x)):   

        amp_erp_all[:, num_sub]  = np.array(list(mod_depth[str(num_sub)][str(i)].values()))
        phase_erp_all[:, num_sub]  = np.array(list(phi[str(num_sub)][str(i)].values()))
        


    amp_array = np.mean(amp_erp_all, axis = 1 )
    phase_array = np.mean(phase_erp_all, axis = 1 )
    amp[str(i)] = amp_array
    phase[str(i)] = phase_array
    



amp_df = pd.DataFrame(amp)
amp_df_array = amp_df.to_numpy()
phi_df = pd.DataFrame(phase)
phi_array = phi_df.to_numpy()
phi_array_deg = np.zeros([len(freq_band), 2])

for i in range(len(ERP_indexs)):
    phi_array_deg[:,i] =  np.degrees(phi_array[:,i])
    for j,j2 in enumerate(freq_band):
        print(j,j2)
        print(j)
        if  phi_array_deg[j,i] < 0:
            phi_array_deg[j,i] =  phi_array_deg[j,i] + 360
        else:
            phi_array_deg[j,i]= phi_array_deg[j,i]
            
            
     
     

    
for i in range(len(ERP_indexs)): 
    cor,ci = pycircstat.corrcc(np.array(freq_band), phi_array_deg[:,i], ci=True)
    cor = np.abs(cor)
    rval=str(np.round(cor,3))
    tval = (cor*(np.sqrt(len(np.array(freq_band)-2)))/(np.sqrt(1-cor**2)))
    pval= str(np.round(1-stats.t.cdf(np.abs(tval),len(np.array(freq_band))-1),3))
    # plot scatter
 

    im = ax[i].scatter(phi_array_deg[:,i], freq_band, c= amp_df_array[:,i])    
    if i==0:
        erp_num = 'First'
    else:
        erp_num = 'Second'
    ax[i].title.set_text(f'{erp_num} ERP, r = {rval}, p = {pval}' )
    clb = fig.colorbar(im, ax=ax[i])    
    clb.ax.set_title('Strength of MD')
    fig.suptitle('Real Time, per subject')
    ax[i].set_xlim([-20, 400])
    ax[i].set_xlabel('Optimal phases (deg)')
    ax[i].set_ylabel('Frequency (Hz)')


#%% Modulation Depth bar plot




from pandas.plotting import table





amp = {}
p_ind = {}


a = np.zeros([len(freq_band), len(x)])
p = np.zeros([len(freq_band), len(x)])

for i in range(len(ERP_indexs)):

    for num_sub in range(len(x)):  
        
        a[:, num_sub] = np.array(list(mod_depth[str(num_sub)][str(i)].values()))

        p[:, num_sub] = np.array(list(mod_depth[str(num_sub)][str(i)].values()))

    amp[str(i)] = np.mean(a, axis =1)
    p_ind[str(i)] = np.mean(p, axis =1)




amp_df = pd.DataFrame(amp)
amp_df_rename = amp_df.rename( columns={'0': 'ERP_1', '1': 'ERP_2'})
p_val_df = pd.DataFrame(p_ind)
p_val_df_rename = p_val_df.rename( columns={'0': 'ERP_1', '1': 'ERP_2'})
   





for i in np.arange(len(amp_df)):
    print(f'{amp_df.index[i]}' , f'{freq_band[i]} hz')
    amp_df_rename = amp_df_rename.rename(index = {amp_df.index[i] : f'{freq_band[i]} hz'})


amp_df_r = amp_df_rename.T




fig, ax = plt.subplots(1, 1)
amp_df_r.plot(kind="bar", alpha=0.75, rot=0,  colormap = 'viridis_r', title = 'Strength of Modulation Depth, Real-Time, per subject', ax= ax).legend(loc = 'lower right')



table6  =table(ax, np.round(amp_df_r.T, 2), loc="upper right");
table6.set_fontsize(12)
table6.scale(0.5,0.5)
ax.set_ylim(bottom=-0.1, top=1.5)
#ax.grid(False)




    
    
