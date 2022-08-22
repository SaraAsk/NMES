#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 13:13:05 2022

@author: sara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-



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



def fig_2c_plot(erp_amplitude, freq_band, cosinefit, subject_info, freq_step_i, save_folder):
    
    # Fig 2.c
    from pandas.plotting import table
    for i in range(len(erp_amplitude)):
        mod_depth = {}
        surr = {}
        p_val = {}
    
        for jf, freq in enumerate(freq_band):   
            
           mod_depth[str(freq)] = cosinefit[str(i)][str(freq)][0]['amp']
           surr[str(freq)] = cosinefit[str(i)][str(freq)][0]['surrogate']
           p_val[str(freq)] = cosinefit[str(i)][str(freq)][0]['p']
           
        p_val_df = pd.DataFrame({'4Hz': p_val[str(4)], '8Hz': p_val[str(8)], '12Hz': p_val[str(12)], '16Hz': p_val[str(16)], \
                                '20Hz': p_val[str(20)], '24Hz': p_val[str(24)], '28Hz': p_val[str(28)], '32Hz': p_val[str(32)], \
                                '36Hz': p_val[str(36)], '40Hz': p_val[str(40)]}, index=['P value'])   
        # Fig2.C
        #mod_depth
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.array(freq_band), np.array(list(mod_depth.values())) , 'k', label='Real data')
        # mod_depth, mod_depth+sd, mod_depth-sd 
        ax.fill_between(np.array(freq_band), np.array(list(mod_depth.values())) + np.std(np.array(list(mod_depth.values()))) , np.array(list(mod_depth.values())) - np.std(np.array(list(mod_depth.values()))),  alpha = 0.6, color = '0.8')
    
    
        #surrogate
        ax.plot(np.array(freq_band), np.mean(np.array(list(surr.values())), axis = 1) , 'r', label='Surrogate')
        ax.fill_between(np.array(freq_band), np.mean(np.array(list(surr.values())), axis = 1) + np.std(np.array(list(surr.values())), axis = 1), np.mean(np.array(list(surr.values())), axis = 1) - np.std(np.array(list(surr.values())), axis = 1), alpha = 0.18, color = 'r')
    
        table6  =table(ax, np.round(p_val_df, 3), loc="upper right", colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        table6.set_fontsize(40)
        table6.scale(0.9,0.9)
        ax.set_ylim(bottom=-0.1, top=2)
        ax.set_xlabel("Frequecies")
        ax.set_ylabel("Strength of Mod")
        ax.legend(loc='lower left')

        
    
        if i==0:
             plt.title(f'1st ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
        else: 
             plt.title(f'2nd ERP, Subject: {subject_info, freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')

    return fig










#%% Group average

import phase_analysis_function as ph_analysis
from scipy.stats import  zscore
from mne import create_info, EpochsArray
from pathlib import Path


exdir_epoch_GA = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
files_GA = Path(exdir_epoch_GA).glob('*epo.fif*')
# =============================================================================
# subjects_with_incomplete_dict =[]
# 
# for f_GA in files_GA:
#     epochs_eeg = mne.read_epochs(f_GA, preload=True)
#     # So basically the problem was mne creats a dict of all stimulation conditions in our case 80. For some epochs data with a small
#     # size all these 80 conditions are not present. It can be 76 so the dict will start from zero to 76 and event_id keys and value will be 
#     # different for each condition in different subjects and there will be a problem during concatinating.
#     # I created a diffault dict, based on 80 condition and forced it to be the same for other epoch files even for the one with less
#     # than 80 conditions.
#     if len(epochs_eeg.event_id) < 80:
#         
#         print(f_GA.parts[-1][0:9])
#         subjects_with_incomplete_dict.append(f_GA.parts[-1][0:9])
# =============================================================================


epochs_eeg, all_channels_clustered, ERP1_chan, ERP2_chan = ph_analysis.epoch_concat_clustered_and_mod_dict(files_GA)

save_folder = "/home/sara/NMES/analyzed data/phase_analysis/Figures/"




    
    
labels, ch_names = ph_analysis.get_ERP_1st_2nd(plot = False)
    


    
# Getting the indices of fisrt and second ERP in epochs_eeg
_, _, ERP1_ch_indx = np.intersect1d( ERP1_chan, epochs_eeg.info['ch_names'], return_indices=True  )
_, _, ERP2_ch_indx = np.intersect1d( ERP2_chan, epochs_eeg.info['ch_names'], return_indices=True  )

ERP1_ch_indx =  np.sort(ERP1_ch_indx)
ERP2_ch_indx =  np.sort(ERP2_ch_indx)
ERP_indexs = [ERP1_ch_indx.T, ERP2_ch_indx.T]

        
  
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
    
    
    sfreq = 1000.0
    if i_ch == 0:
        ch_names = ERP1_chan
    else:
        ch_names = ERP2_chan
        

    
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    
    
    for freq in np.arange(4,44,4):
        epochs_byfreqandphase[str(i_ch)][str(freq)] = {}
        ERP_byfreqandphase[str(i_ch)][str(freq)] = {}
        evoked_zscored[str(i_ch)][str(freq)] = {}
        for phase in np.arange(0,360,45):
            sel_idx = ph_analysis.Select_Epochs(epochs_eeg, freq, phase)
            epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = epochs_eeg[sel_idx]
            
            
            data = epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)]._data[:, ch, labels[i_ch]: labels[i_ch] + 3]
            epochs = EpochsArray(data=data, info=info, events=epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)].events, event_id=epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)].event_id)
            evoked_zscored[str(i_ch)][str(freq)][str(phase)] = np.mean(epochs.average().data)
            
            #ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)] = np.mean(epochs_byfreqandphase[str(i_ch)][str(freq)][str(phase)]._data[:, ch, labels[i_ch]: labels[i_ch] + 3], axis =2)
            #evoked_zscored[str(i_ch)][str(freq)][str(phase)] = np.mean((np.mean(ERP_byfreqandphase[str(i_ch)][str(freq)][str(phase)], axis = 1)))
 
        
cosinefit_ll, amplitudes_cosine_ll, pvalues_cosine_ll = ph_analysis.do_cosine_fit_ll(evoked_zscored, np.arange(0,360,45), np.arange(4,44,4), labels, perm = True)

subject_info = 'Group Average'
    
fig_2c_ll = fig_2c_plot(evoked_zscored, np.arange(4,44,4), cosinefit_ll, subject_info,'Real-time', save_folder)
#fig_2a_ll = ph_analysis.fig_2a_plot(evoked_zscored, np.arange(4,44,4), subject_info,'Real-time', save_folder, vmin = -2, vmax= 2)




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



fig, ax =  plt.subplots(1,2)


freq_band = np.arange(4,44,4)

for i in range(len(ERP_indexs)):
    amp = {}
    phi = {}
    for jf, freq in enumerate(freq_band):   
            
        amp[str(freq)] = cosinefit_ll[str(i)][str(freq)][0]['amp']  
        phi[str(freq)] = cosinefit_ll[str(i)][str(freq)][0]['Fit'].best_values['phi']
    

    
            
    amp_df = pd.DataFrame({'4Hz': amp[str(4)], '8Hz': amp[str(8)], '12Hz': amp[str(12)], '16Hz': amp[str(16)], \
                                '20Hz': amp[str(20)], '24Hz': amp[str(24)], '28Hz': amp[str(28)], '32Hz': amp[str(32)], \
                                '36Hz': amp[str(36)],'40Hz': amp[str(40)]}, index=['amp'])   
    amp_df_array = amp_df.to_numpy()

    phi_df = pd.DataFrame({'4Hz': phi[str(4)], '8Hz': phi[str(8)], '12Hz': phi[str(12)], '16Hz': phi[str(16)], \
                                '20Hz': phi[str(20)], '24Hz': phi[str(24)], '28Hz': phi[str(28)], '32Hz': phi[str(32)], \
                                '36Hz': phi[str(36)], '40Hz': phi[str(40)]}, index=['phi']) 
    phi_array = phi_df.to_numpy()
    phi_array_deg = np.zeros([len(freq_band), 3])


    phi_array_deg[:,i] =  np.degrees(phi_array)
    for j,j2 in enumerate(freq_band):
        print(j,j2)
        print(j)
        if  phi_array_deg[j,i] < 0:
            phi_array_deg[j,i] =  phi_array_deg[j,i] + 360
            
     
     

    

    cor,ci = pycircstat.corrcc(np.array(freq_band), phi_array_deg[:,i], ci=True)
    cor = np.abs(cor)
    rval=str(np.round(cor,4))
    tval = (cor*(np.sqrt(len(np.array(freq_band)-2)))/(np.sqrt(1-cor**2)))
    pval= str(np.round(1-stats.t.cdf(np.abs(tval),len(np.array(freq_band))-1),3))
    # plot scatter
 

    im = ax[i].scatter(phi_array_deg[:,i], freq_band, c= amp_df_array)    
    if i==0:
        erp_num = 'First'
    else:
        erp_num = 'Second'
    ax[i].title.set_text(f'{erp_num} ERP, r = {rval}, p = {pval}' )
    clb = fig.colorbar(im, ax=ax[i])    
    clb.ax.set_title('Strength of MD')
    fig.suptitle('Group Average, Real-Time')
    ax[i].set_xlim([0, 400])
    ax[i].set_xlabel('Optimal phases (deg)')
    ax[i].set_ylabel('Frequency (Hz)')
    ax[i].set_xlim(left=-10)


#%% Modulation depth bar plot
from pandas.plotting import table

mod_depth = {}
p_val = {}
for i in range(len(ERP_indexs)):
    mod_depth[str(i)] = {} 
    p_val[str(i)] = {}

    for jf, freq in enumerate(freq_band):   
        

       mod_depth[str(i)][str(freq)] = cosinefit_ll[str(i)][str(freq)][0]['amp']
       p_val[str(i)][str(freq)] = cosinefit_ll[str(i)][str(freq)][0]['p']
       
p_val_df = pd.DataFrame(p_val)  
        
amp_df = pd.DataFrame(mod_depth)




amp_df_rename = amp_df.rename( columns={'0': 'ERP_1', '1': 'ERP_2'})

p_val_df_rename = p_val_df.rename( columns={'0': 'ERP_1', '1': 'ERP_2'})
    
    
    
    

for i in np.arange(len(amp_df)):
    amp_df_rename = amp_df_rename.rename(index = {f'{amp_df.index[i]}' : f'{amp_df.index[i]} hz'})

amp_df_r = amp_df_rename.T
#amp_df_r.plot(kind="bar", alpha=0.75, rot=0,  colormap = 'viridis_r', title = 'Strength of Modulation Depth').legend(loc = 'upper right')







fig, ax = plt.subplots(1, 1)
amp_df_r.plot(kind="bar", alpha=0.75, rot=0,  colormap = 'viridis_r', title = 'Strength of Modulation Depth, Real-Time, Group Average', ax= ax).legend(loc = 'lower right')



table6  =table(ax, np.round(p_val_df_rename.T, 3), loc="upper right");
table6.set_fontsize(30)
table6.scale(0.9,0.9)
ax.set_ylim(bottom=-0.1, top=2)
ax.grid(False)











