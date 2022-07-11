
"""
Created on Tue Jan 25 10:09:36 2022

@author: sara
"""


import mne
import pickle
import numpy as np
import pandas as pd
from scipy.stats import  zscore
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from mne.stats import permutation_cluster_test


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





def plot_torrecillos_2c(mod_depth, surrogate, i, freq_step_i):
    from textwrap import wrap
    
    if freq_step_i == 0:
        freq_band =  list(range(4, 41, 1))
    elif freq_step_i == 1:
        freq_band = list(range(4, 41, 4))

    amp_erp_all = [] 
    surrogate_erp_all = []  
    for num_sub in range(len(x)):   
        amp_erp_all.append(np.array(list(mod_depth[str(num_sub)][str(freq_step_i)][str(i)].values())))
        surrogate_erp_all.append(np.mean(np.array(list(surrogate[str(num_sub)][str(freq_step_i)][str(i)].values())), axis =1))
        
    amp_erp_all_arr = np.array(amp_erp_all)
    surrogate_erp_all_arr = np.array(surrogate_erp_all)
    
    fig = plt.figure()
    plt.plot(freq_band, np.mean(amp_erp_all_arr, axis = 0 ), color = 'k')
    plt.plot(freq_band, np.mean(surrogate_erp_all_arr, axis = 0 ), color = 'r')
    
    
    plt.fill_between(freq_band, np.std(amp_erp_all_arr, axis = 0 ) + np.mean(amp_erp_all_arr, axis = 0 ),\
                      np.mean(amp_erp_all_arr, axis = 0 ) - np.std(amp_erp_all_arr, axis = 0 ),\
                          color = '0.8')
    
    plt.fill_between(freq_band, np.std(surrogate_erp_all_arr, axis = 0 ) + np.mean(surrogate_erp_all_arr, axis = 0 ),\
                      np.mean(surrogate_erp_all_arr, axis = 0 ) - np.std(surrogate_erp_all_arr, axis = 0 ),\
                          color = 'mistyrose')
    if i==0 and  freq_step_i == 0:
        erp_info= 'ERP1, 1Hz Step'
    elif i==1 and  freq_step_i == 0:
        erp_info= 'ERP2, 1Hz Step' 
    elif i==0 and freq_step_i == 1:
        erp_info= 'ERP1, 4Hz Step'
    elif i==1 and freq_step_i == 1:
        erp_info = 'ERP2, 4Hz Step'
        
    plt.title("\n".join(wrap((f'Group Average: Strength of Modulation {erp_info}'), 38)))
    plt.ylabel('Strength of Mod.') 
    plt.xlabel('Frequencies (Hz)')    
    
    
    threshold =6
    
    # cluster permutation test
    T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([amp_erp_all_arr, surrogate_erp_all_arr], n_permutations=1000,
                             threshold=threshold, tail=1, n_jobs=1,
                             out_type='mask')

    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            h = plt.axvspan(freq_band[c.start], freq_band[c.stop - 1],
                            color='r', alpha=0.3)
        else:
            plt.axvspan(freq_band[c.start], freq_band[c.stop - 1], color=(0.3, 0.3, 0.3),
                        alpha=0.3)
    

    #plt.plot(freq_band, T_obs, 'g')
    plt.legend((h, ), ('cluster p-value < 0.05', ))
        
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
save_folder =  "/home/sara/NMES/analyzed data/phase_analysis"
save_folder_pickle =  "/home/sara/NMES/analyzed data/phase_analysis/pickle/"
save_folder_fig = "/home/sara/NMES/analyzed data/phase_analysis/Figures/cluster_freq/each_subject"

amplitudes_cosines_all_subjects = []
amplitudes_cosines_all_subjects_LL = []
all_subjects_names = []

cosine_fit_all_subjects = []
cosine_fit_all_subjects_LL = []


# Name of the clustered channels must be before looping though epochs of subject, so this process only happens one time. 
all_channels_clustered, ERP1_chan, ERP2_chan = ph_analysis.clustering_channels()    

for f in files:

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
    epochs_bip.save(save_folder +'/epochs bipolar/' + 'epochs_'+ str(f.parts[-3] +'_' + str(f.parts[-1][39:43] )  ) +'_epo.fif' , overwrite = True, split_size='2GB')

    # Step 3: extractinf phase and frequency. Functions in this step are written by Johanna.
    # An array, rows: frequencies of interests. columns: number of epochs. values in this array are phases.
    bin_num = 8 # So it will be consistent with online target phases
    freq_steps = ['1Hz step', '4Hz step']
    amplitudes_cosines = []
    cosine_fit_subjects = []
    
    for freq_step_i in freq_steps:
        print(freq_step_i)
        epochs_bip_fil_phase, target_freq = ph_analysis.extract_phase(epochs_bip._data[: , 1, :], 1000, freq_step_i, 2, angletype = 'degree')
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
        if freq_step_i == '4Hz step':
            freq_band =  list(range(4, 41, 4))
        elif freq_step_i == '1Hz step':
            freq_band = list(range(4, 41))
            
        
                

            
        labels, ch_names = ph_analysis.get_ERP_1st_2nd(plot = False)
        
        
    
        #print(f.parts)
        epochs_eeg = mne.read_epochs(files_epoch_name, preload=True).copy().pick_types(eeg=True)
        
                # Getting the indices of fisrt and second ERP in epochs_eeg
        _, _, ERP1_ch_indx = np.intersect1d( ERP1_chan,ch_names, return_indices=True  )
        _, _, ERP2_ch_indx = np.intersect1d( ERP2_chan, ch_names, return_indices=True  )
        ERP1_ch_indx =  np.sort(ERP1_ch_indx)
        ERP2_ch_indx =  np.sort(ERP2_ch_indx)
        ERP_indexs = [ERP1_ch_indx.T, ERP2_ch_indx.T]
        
        # Choosing the epochs of the bipolar signals accoeding to epochs of EEG channels. So the rejected epochs won't be counted. 
        epochs_num_eeg = epochs_eeg.selection
        epochs_num_bip = epochs_bip.selection
        epochs_num_bip_eeg, epochs_bip_ind,_ = np.intersect1d(epochs_num_bip, epochs_num_eeg, return_indices=True)
        eeg_epoch_filt_phase, target_freq = ph_analysis.extract_phase(epochs_bip._data[epochs_bip_ind , 1, :], 1000, freq_step_i, 2, angletype = 'degree')
        
        bin_class = ph_analysis.assign_bin_class(eeg_epoch_filt_phase, bin_num = bin_num)
    
        erp_amplitude = {}
        for i_ch, ch in enumerate(ERP_indexs):
            # Finding erp amplitudes
            labels_eeg_epoch_freq_ph_bin = {}
            erp_amplitude[str(i_ch)] = {}
            
            for idx, value_freq in enumerate(freq_band):    
                labels_eeg_epoch_freq_ph_bin[value_freq] = {}
                erp_amplitude[str(i_ch)][value_freq] = {}
                
                for value_bin in list(range(0,bin_num)):           
                    labels_eeg_epoch_freq_ph_bin[value_freq][value_bin] = np.where(bin_class[idx,:] == value_bin)[0]    
                    # 2nd mean is for avg across clustered channels, 1st mean avg over epochs      
                    erp_amplitude[str(i_ch)][value_freq][value_bin] = np.mean(np.mean(epochs_eeg._data[labels_eeg_epoch_freq_ph_bin[value_freq][value_bin]][:, ch, labels[i_ch]], axis = 1)) 
                    
                
                
            
        
    
    #%% Fitting the cosine fit to ERP amplitudes, Torrecillos 2020, Fig 2
        perm=True
        cosinefit = {}
        pvalues_cosine = {}
        amplitudes_cosine = {}
        print(freq_step_i)
        cosinefit[freq_step_i], amplitudes_cosine[freq_step_i], pvalues_cosine[freq_step_i] = ph_analysis.do_cosine_fit(erp_amplitude, phase_bin_means, freq_band, labels, perm = True)
        
        
        fig_2c = fig_2c_plot(erp_amplitude, freq_band, cosinefit[freq_step_i], subject_info, freq_step_i, save_folder_fig)
        fig_2a = fig_2a_plot(erp_amplitude, freq_band, subject_info, freq_step_i, save_folder_fig, vmin = -1, vmax= 1)
       
        amplitudes_cosines.append(amplitudes_cosine[freq_step_i])
        cosine_fit_subjects.append(cosinefit[freq_step_i])
        
    amplitudes_cosines_all_subjects.append(amplitudes_cosines)
    cosine_fit_all_subjects.append(cosine_fit_subjects)
    all_subjects_names.append(f.parts[-3] + '_' + f.parts[-1][-8:-4]) 

        #fig_2c.savefig(save_folder + 'fig_2c' + '_' + str(subject_info[-3]) + '_' + str(freq_step_i) + '.png')
        #fig_2a.savefig(save_folder + 'fig_2a' + '_' + str(subject_info[-3]) + '_' + str(freq_step_i) + '.png')

        
         

#%%


    
    
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
                sel_idx = ph_analysis.Select_Epochs(epochs_eeg, freq, phase)
                epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = epochs_eeg[sel_idx]
                ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)]._data[:, ch, labels[i_ch]]
                evoked_zscored[str(i_ch)][str(freq)][str(phase)] = np.mean(np.mean(ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)], axis = 1))
                if str(evoked_zscored[str(i_ch)][str(freq)][str(phase)]) == 'nan':
                    evoked_zscored[str(i_ch)][str(freq)][str(phase)] = 0
            
    cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll = ph_analysis.do_cosine_fit_ll(evoked_zscored, np.arange(0,360,45), np.arange(4,44,4), labels, perm = True)
    
    amplitudes_cosines_all_subjects_LL.append(amplitudes_cosine_ll)
    cosine_fit_all_subjects_LL.append(cosinefit_ll)
    
    if not (cosinefit_ll[str(0)] ):
        print(f'There are not enough epochs by freq and phase for Subject: {subject_info[-3]}')
    else:
        
        fig_2c_ll = fig_2c_plot(evoked_zscored, np.arange(4,44,4), cosinefit_ll, subject_info, 'Real-time', save_folder_fig)
        fig_2a_ll = fig_2a_plot(evoked_zscored, np.arange(4,44,4),  subject_info, 'Real-time', save_folder_fig,  vmin = -2, vmax= 2)
        
        fig_2c_ll.savefig(save_folder_fig + 'fig_2c' + '_' + str(subject_info[-3]) + '_' + 'Real-time' + '.png')
        fig_2a_ll.savefig(save_folder_fig + 'fig_2a' + '_' + str(subject_info[-3]) + '_' + 'Real-time' + '.png')
        
        
        # Fig 2.c but with a comparison of 1Hz, 4Hz steps and lucky loop labels
        for i in range(len(ERP_indexs)):
            if i == 0:
                ERP = '1st ERP'
            else:
                ERP = '2nd ERP'
            fig_mod_comparison =  plt.figure()
            plt.plot(np.arange(4, 44, 4), amplitudes_cosine_ll[:,i], 'r',   label = 'Luck Loop Labels')
            plt.plot(np.arange(4, 44, 4), amplitudes_cosines[1][:, i], 'k', label = '4Hz Steps')
            plt.plot(np.arange(4, 41, 1), amplitudes_cosines[0][:, i], 'b', label = '1Hz Steps')
            plt.title(f' Subject: {subject_info[-3], ERP}')
            plt.ylabel('Strenght of Mod')
            plt.xlabel('Frequencies (Hz)')
            plt.legend()
    
    
            fig_mod_comparison.savefig(save_folder_fig + 'fig_mod_comparison' + '_' + str(subject_info[-3]) + '_' + str(ERP) + '.png')
    


# Saving the pickle files and plotting the strength of Mod by the average of subjects

names = 'all_subjects_names'+ '.p'
with open(str(save_folder_pickle) + names, 'wb') as fp:
    pickle.dump(all_subjects_names, fp, protocol=pickle.HIGHEST_PROTOCOL)
   
cosine_amp_LL = 'amplitudes_cosines_all_subjects_LL'+'.p'
with open(str(save_folder_pickle) + cosine_amp_LL, 'wb') as fp:
    pickle.dump(amplitudes_cosines_all_subjects_LL, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
cosine_amp = 'amplitudes_cosines_all_subjects'+'.p'
with open(str(save_folder_pickle) + cosine_amp, 'wb') as fp:
    pickle.dump(amplitudes_cosines_all_subjects, fp, protocol=pickle.HIGHEST_PROTOCOL)    
 
  
cosine_fit = 'cosine_fit_all_subjects' + '.p'
with open(str(save_folder_pickle) + cosine_fit, 'wb') as fp:
    pickle.dump(cosine_fit_all_subjects, fp, protocol=pickle.HIGHEST_PROTOCOL)    
    
cosine_fit_ll = 'cosine_fit_all_subjects_ll' + '.p'
with open(str(save_folder_pickle) + cosine_fit, 'wb') as fp:
    pickle.dump(cosine_fit_all_subjects_LL, fp, protocol=pickle.HIGHEST_PROTOCOL)        
    








# Plotting AVG of mod depth
  
cos_erp_1_1hz = [] 
cos_erp_2_1hz = []
for i in range(len(amplitudes_cosines_all_subjects)):
    cos_erp_1_1hz.append((amplitudes_cosines_all_subjects[i][0][:,0]))
    cos_erp_2_1hz.append((amplitudes_cosines_all_subjects[i][0][:,1]))
cos_erp_1_mean_1hz = np.mean(cos_erp_1_1hz, axis  = 0)
cos_erp_2_mean_1hz = np.mean(cos_erp_2_1hz, axis  = 0)        

cos_erp_1_4hz = [] 
cos_erp_2_4hz = []
for i in range(len(amplitudes_cosines_all_subjects)):
    cos_erp_1_4hz.append((amplitudes_cosines_all_subjects[i][1][:,0]))
    cos_erp_2_4hz.append((amplitudes_cosines_all_subjects[i][1][:,1]))    
cos_erp_1_mean_4hz = np.mean(cos_erp_1_4hz, axis  = 0)
cos_erp_2_mean_4hz = np.mean(cos_erp_2_4hz, axis  = 0)           

avg_1 = plt.figure()    
plt.title('ERP 1,Avg per Subject')
plt.plot(np.arange(0, 37), cos_erp_1_mean_1hz, label = '1Hz Step')
plt.plot(np.arange(0, 37, 4),cos_erp_1_mean_4hz, label = '4Hz Step')
plt.ylim(0.4,1)
plt.legend()
plt.ylabel('Strenght of Mod')
plt.xlabel('Frequencies (Hz)')
avg_1.savefig(save_folder_fig + 'Avg per Subject' + '_' + 'ERP1' + '.png')
avg_2 = plt.figure()
plt.title('ERP 2, Avg Subject')
plt.plot(np.arange(0, 37), cos_erp_2_mean_1hz, label = '1Hz Step')
plt.plot(np.arange(0, 37, 4),cos_erp_2_mean_4hz, label = '4Hz Step')
plt.ylim(0.4,1)
plt.legend()
plt.ylabel('Strenght of Mod')
plt.xlabel('Frequencies (Hz)')
avg_2.savefig(save_folder_fig + 'Avg per Subject' + '_' + 'ERP2' + '.png')
#%%

with open('/home/sara/NMES/analyzed data/phase_analysis/pickle/cosine_fit_all_subjects.p', 'rb') as f:
    x = pickle.load(f)
    
    
# We have 21 subjects in totals, but for some of them these xdf files is being recorded seperately.   

freq_steps = ['1Hz step', '4Hz step'] 
mod_depth = {}
surrogate = {}
phi = {}
mag = {}

for num_sub in range(len(x)):
    mod_depth[str(num_sub)] = {}
    surrogate[str(num_sub)] = {}
    phi[str(num_sub)] = {}
    mag[str(num_sub)] = {}
    
    for freq_step_i, freq_step_hz in enumerate(freq_steps): 
        mod_depth[str(num_sub)][str(freq_step_i)] = {}
        surrogate[str(num_sub)][str(freq_step_i)] = {}
        phi[str(num_sub)][str(freq_step_i)] = {}
        mag[str(num_sub)][str(freq_step_i)] = {}

        if freq_step_hz == '4Hz step':
            freq_band =  list(range(4, 41, 4))
        elif freq_step_hz == '1Hz step':
            freq_band = list(range(4, 41))

        
        for i in range(2): # len erp amplitude
            mod_depth[str(num_sub)][str(freq_step_i)][str(i)] = {}
            surrogate[str(num_sub)][str(freq_step_i)][str(i)]  = {}
            phi[str(num_sub)][str(freq_step_i)][str(i)]  = {}
            mag[str(num_sub)][str(freq_step_i)][str(i)]  = {}

            for jf, freq in enumerate(freq_band):  
                mod_depth[str(num_sub)][str(freq_step_i)][str(i)][str(freq)] = x[num_sub][freq_step_i][str(i)][str(freq)][0]['amp']
                surrogate[str(num_sub)][str(freq_step_i)][str(i)][str(freq)] = x[num_sub][freq_step_i][str(i)][str(freq)][0]['surrogate']
                phi[str(num_sub)][str(freq_step_i)][str(i)][str(freq)] = x[num_sub][freq_step_i][str(i)][str(freq)][0]['Fit'].best_values['phi']
                mag[str(num_sub)][str(freq_step_i)][str(i)][str(freq)] = x[num_sub][freq_step_i][str(i)][str(freq)][0]['Fit'].best_fit
                



# i= 0 ; freq_step_i = 0 # ERP1, 1Hz step

plot_torrecillos_2c(mod_depth, surrogate, 0, 0) # ERP1, 1Hz step
plot_torrecillos_2c(mod_depth, surrogate, 1, 0) # ERP2, 1Hz step
plot_torrecillos_2c(mod_depth, surrogate, 0, 1) # ERP1, 1Hz step
plot_torrecillos_2c(mod_depth, surrogate, 1, 1) # ERP2, 1Hz step
    

        
#%% Optimal phase distributation

 
# load 
all_channels_clustered, ERP1_chan, ERP2_chan = ph_analysis.clustering_channels()    
labels, ch_names = ph_analysis.get_ERP_1st_2nd(plot = False)

        # Getting the indices of fisrt and second ERP in epochs_eeg
_, _, ERP1_ch_indx = np.intersect1d( ERP1_chan, ch_names, return_indices=True  )
_, _, ERP2_ch_indx = np.intersect1d( ERP2_chan, ch_names, return_indices=True  )
ERP1_ch_indx =  np.sort(ERP1_ch_indx)
ERP2_ch_indx =  np.sort(ERP2_ch_indx)
ERP_indexs = [ERP1_ch_indx.T, ERP2_ch_indx.T]


    

# I just plot it for 4 hz step
mag_erp = []
for i in range(len(ERP_indexs)):

    mag_all = [] 
    for num_sub in range(len(x)):   
        mag_all.append(np.array(list(mag[str(num_sub)][str(1)][str(i)].values())))

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





freq_steps = ['1Hz step', '4Hz step']


for freq_step_i, freq_step in enumerate(freq_steps):
    print(freq_step_i, freq_step)

    if freq_step == '4Hz step':
        freq_band =  list(range(4, 41, 4))
    elif freq_step == '1Hz step':
        freq_band = list(range(4, 41))
    print(freq_step_i, freq_step)



    amp_erp_all = np.zeros([len(freq_band), len(x)])
    phase_erp_all = np.zeros([len(freq_band), len(x)])
    
    fig, ax =  plt.subplots(1,2)
    
    amp = {}
    phase = {}
    for i in range(len(ERP_indexs)):
        amp[str(i)] = {}
        phase[str(i)] = {}
    
        for num_sub in range(len(x)):   
    
            amp_erp_all[:, num_sub]  = np.array(list(mod_depth[str(num_sub)][str(freq_step_i)][str(i)].values()))
            phase_erp_all[:, num_sub]  = np.array(list(phi[str(num_sub)][str(freq_step_i)][str(i)].values()))
            


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
        fig.suptitle(freq_step)
        ax[i].set_xlim([-20, 400])
        ax[i].set_xlabel('Optimal phases (deg)')
        ax[i].set_ylabel('Frequency (Hz)')


#%% Modulation Depth bar plot





freq_steps = ['1Hz step', '4Hz step']



for freq_step_i, freq_step in enumerate(freq_steps):
    print(freq_step_i, freq_step)

    if freq_step == '4Hz step':
        freq_band =  list(range(4, 41, 4))
    elif freq_step == '1Hz step':
        freq_band = list(range(4, 41))


    amp = {}
    p_ind = {}

    
    a = np.zeros([len(freq_band), len(x)])
    p = np.zeros([len(freq_band), len(x)])
    
    for i in range(len(ERP_indexs)):

        for num_sub in range(len(x)):  
            
            a[:, num_sub] = np.array(list(mod_depth[str(num_sub)][str(freq_step_i)][str(i)].values()))
    
            p[:, num_sub] = np.array(list(mod_depth[str(num_sub)][str(freq_step_i)][str(i)].values()))
    
        amp[str(i)] = np.mean(a, axis =1)
        p_ind[str(i)] = np.mean(p, axis =1)




    amp_df = pd.DataFrame(amp)
    amp_df_rename = amp_df.rename( columns={'0': 'ERP_1', '1': 'ERP_2'})
    
   
    
    
    
    

    for i in np.arange(len(amp_df)):
        print(f'{amp_df.index[i]}' , f'{freq_band[i]} hz')
        amp_df_rename = amp_df_rename.rename(index = {amp_df.index[i] : f'{freq_band[i]} hz'})
    

    amp_df_r = amp_df_rename.T
    amp_df_r.plot(kind="bar", alpha=0.75, rot=0,  colormap = 'viridis_r', title = 'Strength of Modulation Depth').legend(loc = 'upper right')


    plt.text(-0.13, 0.8,"*",ha='center',fontsize=12)
    plt.text(-0.07, 0.8,"*",ha='center',fontsize=12)
    plt.text(-0.02, 0.8,"*",ha='center',fontsize=12)
    
    
#%% Cluster permutation test against zero

erp_1_mod_amp = [cos_erp_1_1hz, cos_erp_1_4hz]

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 4))    

for mod_amp in erp_1_mod_amp:
    
    condition1 = pd.DataFrame(mod_amp).to_numpy()
    condition2 =  np.zeros(np.shape(condition1))

    # This part is for 
    if np.shape(condition1)[1] == 10:
        freq_band =  list(range(4, 41, 4))
        threshold = 130
        i = 0
    elif  np.shape(condition1)[1] == 37:
        freq_band = list(range(4, 41))
        threshold = 165
        i = 1
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([condition1, condition2], n_permutations=1000,
                                 threshold=threshold, tail=1, n_jobs=1,
                                 out_type='mask')
    
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            h = axs[i].axvspan(freq_band[c.start], freq_band[c.stop - 1],
                            color='r', alpha=0.3)
        else:
            axs[i].axvspan(freq_band[c.start], freq_band[c.stop - 1], color=(0.3, 0.3, 0.3),
                        alpha=0.3)
    
    
    hf = axs[i].plot(freq_band, T_obs, 'g')
    axs[i].legend((h, ), ('cluster p-value < 0.05', ))
    axs[i].set_xlabel("time (ms)")
    axs[i].set_ylabel("f-values")   
axs[0].set_title('ERP1 4Hz step')
axs[1].set_title('ERP1 1Hz step')    


#############################################


erp_2_mod_amp = [cos_erp_2_1hz, cos_erp_2_4hz]

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 4))    

for mod_amp in erp_2_mod_amp:
    
    condition1 = pd.DataFrame(mod_amp).to_numpy()
    condition2 =  np.zeros(np.shape(condition1))

    # This part is for 
    if np.shape(condition1)[1] == 10:
        freq_band =  list(range(4, 41, 4))
        threshold = 200
        i = 0
    elif  np.shape(condition1)[1] == 37:
        freq_band = list(range(4, 41))
        threshold = 160
        i = 1
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([condition1, condition2], n_permutations=1000,
                                 threshold=threshold, tail=1, n_jobs=1,
                                 out_type='mask')
    
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            h = axs[i].axvspan(freq_band[c.start], freq_band[c.stop - 1],
                            color='r', alpha=0.3)
        else:
            axs[i].axvspan(freq_band[c.start], freq_band[c.stop - 1], color=(0.3, 0.3, 0.3),
                        alpha=0.3)
    
    
    hf = axs[i].plot(freq_band, T_obs, 'g')
    axs[i].legend((h, ), ('cluster p-value < 0.05', ))
    axs[i].set_xlabel("time (ms)")
    axs[i].set_ylabel("f-values")   
axs[0].set_title('ERP2 4Hz step')
axs[1].set_title('ERP2 1Hz step')    


#%%




