#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:52:39 2022

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
        surr_std ={}
        surr = {}
        for jf, freq in enumerate(freq_band):   
           mod_depth[str(freq)] = cosinefit[str(i)][str(freq)][0]['amp']

           surr[str(freq)] = cosinefit[str(i)][str(freq)][0]['surrogate']
           surr_std[str(freq)] = np.std(surr[str(freq)], axis =0)

           # Fig2.C
        fig = plt.figure()
        plt.plot(np.array(freq_band), (np.array(list(mod_depth.values()))), 'k', label='Real data')
        plt.plot(np.array(freq_band), np.mean((np.array(list(surr.values()))), axis = 1), 'r', label='Surrogate')
        np.array(list(surr.values()))
        

        
        plt.fill_between(freq_band, np.array(list(surr_std.values())) + np.mean( np.array(list(surr.values())), axis = 1 ),\
                         np.mean( np.array(list(surr.values())), axis = 1 ) - np.array(list(surr_std.values())),\
                           alpha = 0.18, color = 'r')
        plt.xlabel("Frequecies")
        plt.ylabel("Strength of Mod")
        plt.legend(loc='lower right')

        if i==0:
             plt.title(f'1st ERP, Subject: {subject_info[-1][0:9], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-1][0:9]) + '_'+ '1st ERP' +'_' + str(freq_step_i) + '.png')
             
        else: 
             plt.title(f'2nd ERP, Subject: {subject_info[-1][0:9], freq_step_i}')
             fig.savefig(save_folder + 'fig_2c_all_cluster' + '_' + str(subject_info[-1][0:9]) + '_'+ '2nd ERP' +'_' + str(freq_step_i) + '.png')
    
    return fig





def plot_torrecillos_2c(mod_depth, surrogate, i):
    from textwrap import wrap
    from pandas.plotting import table
   

    amp_erp_all = [] 
    surrogate_erp_all = []  
    for num_sub in range(len(x)):  
        print(num_sub)
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
    ax.legend(loc='lower right')
    ax.set_ylim(bottom=-0.1, top=1.7)

    
    threshold = 2.6
    

    
    
    # cluster permutation test
    T_obs, clusters, cluster_p_values, H0 = \
    permutation_cluster_test([amp_erp_all_arr, surrogate_erp_all_arr], n_permutations=1000,
                             threshold=threshold, tail=1, n_jobs=1,
                             out_type='mask')

    for i_c, c in enumerate(clusters):
        print(i_c, c)
     
        c = c[0]
        if cluster_p_values[i_c] <= 0.05 and (freq_band[c.stop - 1] - freq_band[c.start]) > 0:
            h = plt.axvspan(freq_band[c.start], freq_band[c.stop - 1], ymin = 0.2, ymax = 0.7, 
                            color='g', alpha=0.25)
            print(cluster_p_values[i_c], freq_band[c.start], freq_band[c.stop - 1])
            ax.text( freq_band[c.start]-1, 1.4, f'P-value = {cluster_p_values[i_c]}', color='k') 

            
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
import os
import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

# phase analysis function
import phase_analysis_function as ph_analysis






#%% Extracting phase and frequency from the bipolar channel around C3.

# =============================================================================
# concat_epoch_sub = True
# exdir_epoch = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
# save_folder = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
# 
# 
# files = list(Path(exdir_epoch).glob('*ScSe*'))
# if concat_epoch_sub == True:
# 
#     epoch_concat_subs, info = epoch_concat_subs_mutltiple_files(files)
#     epoch_concat_subs.save(save_folder+ str(info[-1][0:4]) + '_' + 'concat_manually' + '_epo.fif', overwrite = True, split_size='2GB')
#     
# 
# 
# 
# =============================================================================




exdir_epoch = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"



files = list(Path(exdir_epoch).glob('**/*epo.fif'))
#files = list(Path(exdir_epoch).glob('**/*BrTi*'))
save_folder =  "/home/sara/NMES/analyzed data/phase_analysis"
save_folder_pickle =  "/home/sara/NMES/analyzed data/phase_analysis/pickle/"
save_folder_fig = "/home/sara/NMES/analyzed data/phase_analysis/Figures/cluster_freq/per_subject/"

amplitudes_cosines_all_subjects = []
amplitudes_cosines_all_subjects_LL = []
all_subjects_names = []

cosine_fit_all_subjects = []
cosine_fit_all_subjects_LL = []


# Name of the clustered channels must be before looping though epochs of subject, so this process only happens one time. 
all_channels_clustered, ERP1_chan, ERP2_chan, pvals = ph_analysis.clustering_channels()    
labels, ch_names = ph_analysis.get_ERP_1st_2nd(plot = False)
        
        
    

# Getting the indices of fisrt and second ERP in epochs_eeg
_, _, ERP1_ch_indx = np.intersect1d( ERP1_chan,ch_names, return_indices=True  )
_, _, ERP2_ch_indx = np.intersect1d( ERP2_chan, ch_names, return_indices=True  )
ERP1_ch_indx =  np.sort(ERP1_ch_indx)
ERP2_ch_indx =  np.sort(ERP2_ch_indx)
ERP_indexs = [ERP1_ch_indx.T, ERP2_ch_indx.T]



for f in files:
    subject_info = f.parts 
    
    all_subjects_names.append(str(subject_info[-1][0:9]))
    
    #%% Extracting ERP amplitude for frequency and phases according to bipolar channel.
        
    # Subj_path is added to exdir, so the EEG epoch files and bipolar  channels are selected from the same subject. 


    epochs_eeg = mne.read_epochs(f, preload=True).copy().pick_types(eeg=True)
        

        
                

            

    

        
         

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
        #fig_2a_ll = fig_2a_plot(evoked_zscored, np.arange(4,44,4),  subject_info, 'Real-time', save_folder_fig,  vmin = -2, vmax= 2)
        
        #fig_2c_ll.savefig(save_folder_fig + 'fig_2c' + '_' + str(subject_info[-1][0:9]) + '_' + 'Real-time' + '.png')
        #fig_2a_ll.savefig(save_folder_fig + 'fig_2a' + '_' + str(subject_info[-1][0:9]) + '_' + 'Real-time' + '.png')
        
        
    

# Saving the pickle files and plotting the strength of Mod by the average of subjects

names = 'all_subjects_names'+ '.p'
with open(str(save_folder_pickle) + names, 'wb') as fp:
    pickle.dump(all_subjects_names, fp, protocol=pickle.HIGHEST_PROTOCOL)
   
cosine_amp_LL = 'amplitudes_cosines_all_subjects_LL'+'.p'
with open(str(save_folder_pickle) + cosine_amp_LL, 'wb') as fp:
    pickle.dump(amplitudes_cosines_all_subjects_LL, fp, protocol=pickle.HIGHEST_PROTOCOL)
    

    
cosine_fit_ll = 'cosine_fit_all_subjects_ll' + '.p'
with open(str(save_folder_pickle) + cosine_fit_ll, 'wb') as fp:
    pickle.dump(cosine_fit_all_subjects_LL, fp, protocol=pickle.HIGHEST_PROTOCOL)        
    






#%%

with open('/home/sara/NMES/analyzed data/phase_analysis/pickle/cosine_fit_all_subjects_ll.p', 'rb') as f:
    x = pickle.load(f)
  
    
  
with open('/home/sara/NMES/analyzed data/phase_analysis/pickle/all_subjects_names.p', 'rb') as f:
 
    subject_names =  pickle.load(f)     
    
    
    

labels, ch_names = ph_analysis.get_ERP_1st_2nd(plot = False)
    
    
    
    
    
#%%    
    
    
# We have 21 subjects in totals, but for some of them these xdf files is being recorded seperately.   

freq_band = np.arange(4, 44, 4)
mod_depth = {}
surrogate = {}
phi = {}
mag = {}

for num_sub in range(len(x)):
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
            phi[str(num_sub)][str(i)][str(freq)] =  np.degrees(x[num_sub][str(i)][str(freq)][0]['Fit'].best_values['phi'])
            if  np.degrees(x[num_sub][str(i)][str(freq)][0]['Fit'].best_values['phi']) < 0:
                phi[str(num_sub)][str(i)][str(freq)] =  np.degrees(x[num_sub][str(i)][str(freq)][0]['Fit'].best_values['phi']) +360
            mag[str(num_sub)][str(i)][str(freq)] = x[num_sub][str(i)][str(freq)][0]['Fit'].best_fit
            



# i= 0 ; freq_step_i = 0 # ERP1, 1Hz step

plot_torrecillos_2c(mod_depth, surrogate, 0) # ERP1
plot_torrecillos_2c(mod_depth, surrogate, 1) # ERP2

opt_phase = {}
for num_sub in range(len(x)):  
    print(num_sub)
    opt_phase[str(num_sub)] = phi[str(num_sub)][str(1)][str(36)]      

plt.plot(np.array(list(opt_phase.values())))


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
table6.set_fontsize(10)
table6.scale(0.7,0.7)
ax.set_ylim(bottom=-0.1, top=1.5)
ax.grid(False)






