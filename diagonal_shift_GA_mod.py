#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:09:11 2022

@author: sara
"""

import mne
import math
import lmfit
import itertools  
import mne.stats
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










def fig_2a_plot(erp_amplitude, freq_band, cosinefit, freq_step_i, save_folder, vmin = -2, vmax= 2):
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
        
    

    for i in range(len(erp_amplitude)):
        data_erp = {}       
        for jf, freq in enumerate(freq_band):   
            data_erp[str(freq)] = zscore(np.array(list(cosinefit[str(i)][str(freq)][0]['data'].values())))       
        # Heatmaps. 
        fig = plt.figure()
        data_erp_df = pd.DataFrame(data_erp) 
        data_erp_df = data_erp_df.T
        data_erp_arr = zscore(data_erp_df.to_numpy())
        
        # Plotting the biggest cluster
        # Setting a threshold to equal to standard deviation of ERP amplitude
        arr = data_erp_arr > np.std(data_erp_arr)
        arr_largest_cluster = get_largest_component(arr)
        
        
        
        data_erp_df_rename = data_erp_df.rename(columns={0:'0', 1:'45', 2:'90', 3:'135', 4:'180', 5:'225', 6:'270', 7:'315'})
        data_erp_df_ph_reorder = data_erp_df_rename.reindex(columns = ["0", "45", "90", "135", "180", "225", "270", "315"])
        ax = sns.heatmap(data_erp_df_ph_reorder,vmin = vmin, vmax = vmax,   cmap ='viridis')
        
        arr_largest_cluster_ind = np.argwhere(arr_largest_cluster == 1)
        for i in range(len(arr_largest_cluster_ind)):
            ax.add_patch(Rectangle((arr_largest_cluster_ind[i][1], arr_largest_cluster_ind[i][0]),  1, 1,  ec = 'cyan', fc = 'none', lw=2, hatch='//'))
        
        # swap the axes
        ax.invert_yaxis()
        plt.xlabel("Phases", fontsize=16)
        plt.ylabel("Frequencies", fontsize=16)
        # swap the axes
        plt.xlabel("Phases")
        plt.ylabel("Frequencies")
        if i==0:
            plt.title(f'1st ERP, All Subject: {freq_step_i}')
            fig.savefig(save_folder + 'fig_2a' + '_' + 'All_Subjects' + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
        else:
            plt.title(f'2nd ERP, All Subject: {freq_step_i}')
            fig.savefig(save_folder + 'fig_2a' + '_' + 'All_Subjects' + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
    return  fig     


def fig_2c_plot(erp_amplitude, cosinefit_all, save_folder):
    # Fig 2.c
    p_val_arr_all = []
    for freq_step in range(len(cosinefit_all)):
        if freq_step == 0:
            freq_band =  list(range(4, 41, 1))
        else :
            freq_band = list(range(4, 41, 4))

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
            p_val = {}
            for jf, freq in enumerate(freq_band):   

               mod_depth[str(freq)] = cosinefit_all[freq_step][str(i)][str(freq)][0]['amp']
               #data_z[str(freq)] =  np.array(list(cosinefit[str(i)][str(freq)][0]['data'].values()))
               #data_z_min[str(freq)] = min(data_z[str(freq)])
               #data_z_max[str(freq)] = max(data_z[str(freq)])
               best_fit[str(freq)] = cosinefit_all[freq_step][str(i)][str(freq)][0]['Fit'].best_fit
               best_fit_min[str(freq)]  = min(cosinefit_all[freq_step][str(i)][str(freq)][0]['Fit'].best_fit)
               best_fit_max [str(freq)] = max(cosinefit_all[freq_step][str(i)][str(freq)][0]['Fit'].best_fit)
               surr_min[str(freq)] = min(cosinefit_all[freq_step][str(i)][str(freq)][0]['surrogate'])
               surr_max[str(freq)] = max(cosinefit_all[freq_step][str(i)][str(freq)][0]['surrogate'])
               p_val[str(freq)] = cosinefit_all[freq_step][str(i)][str(freq)][0]['p']
               # Fig2.C
           
            p_val_arr = np.array(list(p_val.values()))        
            p_val_arr_all.append(p_val_arr)
            #plt.plot(np.array(freq_band), (np.array(list(best_fit_max.values())) - np.array(list(best_fit_min.values())))/1, 'k') 
            fig = plt.figure()
            plt.plot(np.array(freq_band), np.array(list(mod_depth.values())), 'k')
            #plt.fill_between(np.array(freq_band), (np.array(list(best_fit_max.values())) - np.array(list(best_fit_min.values()))), (np.array(list(best_fit_max.values())) + np.array(list(best_fit_min.values()))),  color = '0.8')
            
            
          
            for p in p_val_arr:
                if p < 0.05:
                    print(p)
                    p_d = round(p, 4)
                    plt.plot(freq_band[np.where(p_val_arr == p)[0][0]],   np.array(list(mod_depth.values()))[np.where(p_val_arr == p)[0][0]]+ 1.3, '*', c ='k', label = f' p = {p_d, freq_band[np.where(p_val_arr == p)[0][0]]} Hz' )
                    plt.legend(loc=1)
            #plt.fill_between(np.array(freq_band), np.array(list(data_z_min.values())), np.array(list(data_z_max.values())), color = '0.8')
            #plt.fill_between(np.array(freq_band), np.array(list(best_fit_min.values())), np.array(list(best_fit_max.values())), color = '0.8')
            #plt.fill_between(np.array(freq_band), np.array(list(surr_min.values())), np.array(list(surr_max.values()))/2, color = 'r')
            #plt.plot(np.array(list(best_fit.values())), 'r')
            plt.xlabel("Frequecies")
            plt.ylabel("Strength of Mod")
      
            if i==0 and freq_step == 0 :
                plt.title('1st ERP, All Subject: 1 Hz Step')
                fig.savefig(save_folder + 'fig_2c' + '_' + 'All_Subjects' + '_'+ '1st ERP' +'_' + '1 Hz Step' + '.png')
            elif i==1 and freq_step == 0 :
                plt.title('2nd ERP, All Subject: 1 Hz Step')
                fig.savefig(save_folder + 'fig_2c' + '_' + 'All_Subjects' + '_'+ '2nd ERP' +'_' + '1 Hz Step' + '.png')
             
            elif i==0 and freq_step == 1 :
                plt.title('1st ERP, All Subject: 4 Hz Step')
                fig.savefig(save_folder + 'fig_2c' + '_' + 'All_Subjects' + '_'+ '1st ERP' +'_' + '4 Hz Step' + '.png')    
            else:
                plt.title(f'2nd ERP, All Subject: {freq_step_i}')
                fig.savefig(save_folder + 'fig_2c' + '_' + 'All_Subjects' + '_' + '2nd ERP' +'_' + '4 Hz Step' + '.png')

    
    return fig











#%% Group average

import phase_analysis_function as ph_analysis
from pathlib import Path
# Bipolar Signals

exdir_epoch_bip = "/home/sara/NMES/analyzed data/phase_analysis/epochs bipolar/"
files_bp = Path(exdir_epoch_bip).glob('*_epo.fif*')
epochs_bip = ph_analysis.epoch_concat_and_mod_dict_bip(files_bp)


exdir_epoch_GA = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
files_GA = Path(exdir_epoch_GA).glob('*epo.fif*')
epochs_eeg, all_channels_clustered, ERP1_chan, ERP2_chan = ph_analysis.epoch_concat_clustered_and_mod_dict(files_GA)

save_folder = "/home/sara/NMES/analyzed data/phase_analysis/Figures/"


freq_steps = ['1Hz step', '4Hz step']
cosinefit_all = []



bin_num = 8 # So it will be consistent with online target phases
amplitudes_cosines = []
for freq_step_i in freq_steps:
    print(freq_step_i)
    epochs_bip_fil_phase, target_freq = ph_analysis.extract_phase(epochs_bip._data[: , 1, :], 1000, freq_step_i, 2, angletype = 'degree')
    #(frequencies of interests, number of epochs). An array like eeg_epoch_filt_phase, but the values are the classes of the phse bins.
    # it says to which class the phases belong to. 
    bin_class = ph_analysis.assign_bin_class(epochs_bip_fil_phase, bin_num = bin_num)
    phase_bin_means = ph_analysis.get_phase_bin_mean(epochs_bip_fil_phase, bin_class, bin_num = bin_num)
    ph_analysis.plot_phase_bins(phase_bin_means, bin_class, bin_num, scaling_proportion = 300)
    #phase_bin_means_per_freq = ph_analysis.get_phase_bin_mean_each_freq(target_freq, epochs_bip_fil_phase, bin_class, bin_num = bin_num)
    #plot_mean_phase_bins_each_freq(phase_bin_means_per_freq, bin_class, bin_num, scaling_proportion =300)
    
    
    
    
    #%%
    
    num_epoch = bin_class.T.shape[0]
    if freq_step_i == '4Hz step':
        freq_band =  list(range(4, 41, 4))
    elif freq_step_i == '1Hz step':
        freq_band = list(range(4, 41))

    
    
    labels, ch_names = ph_analysis.get_ERP_1st_2nd(plot = False)
        


        
    # Getting the indices of fisrt and second ERP in epochs_eeg
    _, _, ERP1_ch_indx = np.intersect1d( ERP1_chan, epochs_eeg.info['ch_names'], return_indices=True  )
    _, _, ERP2_ch_indx = np.intersect1d( ERP2_chan, epochs_eeg.info['ch_names'], return_indices=True  )

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
    #epochs_amplitude = {}
    erp_amplitude_sem = {} # Compute standard error of the mean for error bar after cosine fit.
    for i_ch, ch in enumerate(ERP_indexs):
        print(i_ch, ch)
        # Finding erp amplitudes
        labels_eeg_epoch_freq_ph_bin = {}
        erp_amplitude[str(i_ch)] = {}
        #epochs_amplitude[str(i_ch)] = {}
        erp_amplitude_sem[str(i_ch)] = {}
        
        for idx, value_freq in enumerate(freq_band): 
            print(idx, value_freq)
            labels_eeg_epoch_freq_ph_bin[value_freq] = {}
            erp_amplitude[str(i_ch)][value_freq] = {}
            erp_amplitude_sem[str(i_ch)][value_freq] = {}
            #epochs_amplitude[str(i_ch)][value_freq] = {}
            
            for value_bin in list(range(0,bin_num)):           
                labels_eeg_epoch_freq_ph_bin[value_freq][value_bin] = np.where(bin_class[idx,:] == value_bin)[0]    
                # Choosing the averages over label time to label +10 to be like "cosine_fit_clustring" script 
                # erp_amplitude[str(i_ch)][value_freq][value_bin] = np.mean(epochs_eeg._data[labels_eeg_epoch_freq_ph_bin[value_freq][value_bin]][:, ch, labels[i_ch] : labels[i_ch] + 2], axis = 2)
                # erp_amplitude[str(i_ch)][value_freq][value_bin] = np.mean(np.mean(epochs_eeg._data[labels_eeg_epoch_freq_ph_bin[value_freq][value_bin]][:, ch], axis = 1)) 
                # 2nd mean is for avg across clustered channels, 1st mean avg over epochs 
                
                erp_amplitude[str(i_ch)][value_freq][value_bin] = np.mean(np.mean(epochs_eeg._data[labels_eeg_epoch_freq_ph_bin[value_freq][value_bin]][:, ch, labels[i_ch]], axis = 1)) 
                
                #epochs_amplitude[str(i_ch)][value_freq] = epochs_eeg._data[labels_eeg_epoch_freq_ph_bin[value_freq][value_bin]][:, ch, labels[i_ch]]
                erp_amplitude_sem[str(i_ch)][value_freq][value_bin] = np.std(np.mean(epochs_eeg._data[labels_eeg_epoch_freq_ph_bin[value_freq][value_bin]][:, ch, labels[i_ch]], axis = 1)) /np.sqrt(27) # number of epoch files being concatenated 
                # https://github.com/Fosca/GeomSeq/blob/296b50ade1040d171a5ab0fc1960d396e794bc6b/GeomSeq_functions/ordinal_decoders.py
    
        subject_info = 'Group Average'
        ph_analysis.fig_2a_plot(erp_amplitude, freq_band, subject_info, freq_step_i, save_folder, vmin = -0.7, vmax= 1)
        
        
#%% Fitting the cosine to '4Hz step freq' and plotting for each freq and different phases

    perm=True
    cosinefit = {}
    pvalues_cosine = {}
    amplitudes_cosine = {}
    print(freq_step_i)
    if freq_steps == '4Hz step':
        cosinefit[freq_step_i], amplitudes_cosine[freq_step_i], pvalues_cosine[freq_step_i] = ph_analysis.do_cosine_fit_phase_freq_extracted(erp_amplitude, erp_amplitude_sem, phase_bin_means, freq_band, labels, perm = True)
    


#%% Fitting the cosine fit to ERP amplitudes, Torrecillos 2020, Fig 2

    
    cosinefit[freq_step_i], amplitudes_cosine[freq_step_i], pvalues_cosine[freq_step_i] = ph_analysis.do_cosine_fit(erp_amplitude, phase_bin_means, freq_band, labels, perm = True)
    
    
    #fig_2a = fig_2a_plot(erp_amplitude, freq_band, cosinefit[freq_step_i], freq_step_i, save_folder, vmin = -2, vmax= 2)
   
    amplitudes_cosines.append(amplitudes_cosine[freq_step_i])
    cosinefit_all.append(cosinefit[freq_step_i])


fig_2c = fig_2c_plot(erp_amplitude, cosinefit_all,  save_folder)
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
            ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)]._data[:, ch, labels[i_ch]: labels[i_ch] + 3], axis =2)
            evoked_zscored[str(i_ch)][str(freq)][str(phase)] = np.mean((np.mean(ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)], axis = 1)))
 
        
cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll = ph_analysis.do_cosine_fit_ll(evoked_zscored, np.arange(0,360,45), np.arange(4,44,4), labels, perm = True)

subject_info = 'Group Average'
    
fig_2c_ll = ph_analysis.fig_2c_plot(evoked_zscored, np.arange(4,44,4), cosinefit_ll, subject_info,'Real-time', save_folder)
#fig_2a_ll = ph_analysis.fig_2a_plot(evoked_zscored, np.arange(4,44,4), subject_info,'Real-time', save_folder, vmin = -2, vmax= 2)


#%%
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
    plt.title(f' All Subjects: { ERP}')
    plt.ylabel('Strenght of Mod')
    plt.xlabel('Frequencies (Hz)')
    plt.legend()

    fig_mod_comparison.savefig(save_folder + 'fig_mod_comparison' + '_' + 'All Subjects' + '_' + str(ERP) + '.png')

#%% Optimal phase distribution
# https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
# https://jwalton.info/Matplotlib-rose-plots/
titles = ['Freq 4', 'Freq 8', 'Freq 12', 'Freq 16', 'Freq 20', 'Freq 24', 'Freq 28', 'Freq 32', 'Freq 36', 'Freq 40'] 


mag = {}
for i in range(len(ERP_indexs)):
    mag[str(i)] = {}
    for freq in np.arange(4,44,4):
        mag[str(i)][str(freq)] = {}
        mag[str(i)][str(freq)] = cosinefit_ll[str(i)][str(freq)][0]['Fit'].best_fit


mag_df = pd.DataFrame(mag)
mag_df_array = mag_df.to_numpy()
 


fig = plt.figure(constrained_layout=True)
fig.suptitle('Optimal Phase distribution', fontweight="bold")

# create 3x1 subfigs
subfigs = fig.subfigures(nrows=2, ncols=1)
for row, subfig in enumerate(subfigs):
    subfig.suptitle(f'ERP {row+1}')

    # create 1x3 subplots per subfig
    axs = subfig.subplots(nrows=1, ncols=10,subplot_kw=dict(projection='polar'))
    for col, ax in enumerate(axs):
         ph_analysis.phase_optimal_distribution(ax, mag_df_array[col,row], titles[col])    
 

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

    fig, ax =  plt.subplots(1,2)
    
    amp = {}
    phi = {}
    for i in range(len(ERP_indexs)):
        amp[str(i)] = {}
        phi[str(i)] = {}
        for j, j1 in enumerate(freq_band):
            amp[str(i)][str(j1)] = cosinefit_all[freq_step_i][str(i)][str(j1)][0]['Fit'].best_values['amp']
            phi[str(i)][str(j1)] = cosinefit_all[freq_step_i][str(i)][str(j1)][0]['Fit'].best_values['phi']
            
            
        
                
    amp_df = pd.DataFrame(amp)
    amp_df_array = amp_df.to_numpy()
    phi_df = pd.DataFrame(phi)
    phi_array = phi_df.to_numpy()
    phi_array_deg = np.zeros([len(freq_band), 2])
    
    for i in range(len(ERP_indexs)):
        phi_array_deg[:,i] =  np.degrees(phi_array[:,i])
        for j,j2 in enumerate(freq_band):
            print(j,j2)
            print(j)
            if  phi_array_deg[j,i] < 0:
                phi_array_deg[j,i] =  phi_array_deg[j,i] + 360
                
     
     

    
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
        ax[i].set_xlim([0, 400])
        ax[i].set_xlabel('Optimal phases (deg)')
        ax[i].set_ylabel('Frequency (Hz)')


#%% Modulation depth bar plot



freq_steps = ['1Hz step', '4Hz step']


for freq_step_i, freq_step in enumerate(freq_steps):
    print(freq_step_i, freq_step)

    if freq_step == '4Hz step':
        freq_band =  list(range(4, 41, 4))
    elif freq_step == '1Hz step':
        freq_band = list(range(4, 41))

    
    amp = {}
    p_val = {}
    for i in range(len(ERP_indexs)):
        amp[str(i)] = {}
        p_val[str(i)] = {}
        
        for j, j1 in enumerate(freq_band):
            amp[str(i)][str(j1)] = cosinefit_all[freq_step_i][str(i)][str(j1)][0]['Fit'].best_values['amp']
            p_val[str(i)][str(j1)] = cosinefit_all[freq_step_i][str(i)][str(j1)][0]['p']





    amp_df = pd.DataFrame(amp)
    amp_df_rename = amp_df.rename( columns={'0': 'ERP_1', '1': 'ERP_2'})
    p_val_df = pd.DataFrame(p_val)
    p_val_df_rename = p_val_df.rename( columns={'0': 'ERP_1', '1': 'ERP_2'})
    
    
    
    
    amp_df_rename = amp_df_rename
    for i in np.arange(len(amp_df)):
        amp_df_rename = amp_df_rename.rename(index = {f'{amp_df.index[i]}' : f'{amp_df.index[i]} hz'})
    
    amp_df_r = amp_df_rename.T
    amp_df_r.plot(kind="bar", alpha=0.75, rot=0,  colormap = 'viridis_r', title = 'Strength of Modulation Depth').legend(loc = 'upper right')

