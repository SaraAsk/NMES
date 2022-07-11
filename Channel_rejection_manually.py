#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 09:34:06 2021

@author: sara
"""








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
        fig = plt.figure(20,figsize=[7,5])
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
    return(index_n1[0][0], index_n2[0][0], fig2)



#%%

import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#%%

epochs_data_dir = "/home/sara/NMES/analyzed data/Feb/epochs/"

epochs_files = Path(epochs_data_dir).glob('*LaCa*')
save_folder_epochs = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
save_folder_evokeds = "/home/sara/NMES/analyzed data/Feb/evokeds/"
Save_folder_topo = "/home/sara/NMES/analyzed data/Feb/evokeds/Figures/base_line_corrected/ERP/"
Save_folder_ERP = "/home/sara/NMES/analyzed data//Feb/evokeds/Figures/base_line_corrected/TOPO/"


dictionary_bad_channels = {'BaJo_R001': ['Fp1','Fp2', 'Fpz', 'T8', 'T7', 'FT8'], 'BrTi_R001': ['Fpz', 'Fp1', 'AF7', 'PO7', 'AF4', 'TP9', 'FT9', 'AF8','F7'],
                           'EuSa_R001': ['Fp2', 'Fp1', 'PO7', 'T7', 'T8', 'AF8', 'F8'], 'HaSa_R001' : ['Fpz', 'Fp1', 'PO7', 'Fp2', 'AF7'], 'HaSa_R002':['Fpz', 'TP9', 'PO7', 'Fp2', 'PO4', 'Fp1', 'AF7', 'FC1'], 
                           'HeDo_R010' : ['Fpz', 'Fp2', 'AF8', 'C2', 'C4', 'AF7', 'AF8', 'FT9', 'TP10','Fp1','AF4', 'F8', 'AF8'], 'LaCa_R001': ['Fpz', 'Fp1', 'FT7'], 'LiLu_R001': ['Fp2','AF8','Fpz','FT8','FT10','PO7', 'FC3', 'Fp1'],
                           'MeAm_R001':['PO7','TP7','AF8','F2', 'AF7', 'FT9', 'TP10', 'Fpz', 'Fp2'], 'NeMa_R001':['Fp1', 'AF7','TP9', 'T7', 'TP7'], 'StLa_R001' : ['C1','FC2'],
                           'MiAn_R001':['F1','PO7', 'F3','T8', 'FT8'],'MiAn_R002': ['F3','F1','Fpz','FT10','F5','FC1','PO7'], 'RaLu_R001': ['Fpz','Fp2', 'PO7','P7','TP9'], 
                           'RuMa_R001':['Fp1','Iz', 'CP5', 'TP9', 'FT7', 'FT9'], 'ScSe_R001':['Fpz','Fp1','TP10','Oz','FC3','Iz','F1','O1','C3', 'PO7'], 'ScSe_R002':['Iz','PO7','F1','PO3','TP10','TP9', 'FC3', 'FC5', 'F3','Iz'],
                           'UtLi_R001':['Fpz','Fp2','Iz','PO7','AF8','F8','FT10','AF7','Fpz', 'Iz', 'O1'],'VeVa_R001':['Fp2','Fp1', 'FT10', 'AF4', 'F7', 'AF7', 'P8', 'Iz', 'O1'],'WoIr_R001':['Fp1','AF3','PO8', 'TP9', 'FT9', 'F2'],
                           'ZaHa_R001':['Fpz'],'ZhJi_R001':['TP9'],'ZhJi_R002':['FC1', 'PO3', 'TP9'], 'ZiAm_R001':['Iz','O1','PO7','PO8','Fpz', 'FT10', 'Oz', 'O2']}


for f in epochs_files:
    plt.close('all')
    subject_ID = f.parts[-1][0:9]
    epochs = mne.read_epochs(f, preload= True)
    epochs = epochs.set_eeg_reference(ref_channels='average')
    if subject_ID in dictionary_bad_channels:
       epochs.info['bads'] = dictionary_bad_channels[subject_ID]
       epochs_art1 = epochs.interpolate_bads(reset_bads=True, mode='accurate')



    evokeds_art1 = epochs_art1.average()
    all_times = np.arange(-0.2, 0.5, 0.03)
    topo_plots = evokeds_art1.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
    ERP_plots = evokeds_art1.plot(spatial_colors = True, gfp = True) 
    
    
    #Savings
    topo_plots.savefig(Save_folder_topo + str(f.parts[-1][0:9]) ) 
    ERP_plots.savefig(Save_folder_ERP + str(f.parts[-1][0:9]) ) 
    epochs_art1.save(save_folder_epochs + str(f.parts[-1][0:9]) + '_manually' + '_epo.fif', overwrite = True, split_size='2GB')
    mne.evoked.write_evokeds(save_folder_evokeds + str(f.parts[-1][0:9]) + '_manually' + '_ave.fif', evokeds_art1) 
#%% 

phase_dep_epoch_dir = "/home/sara/NMES/analyzed data/Feb/Epochs_NMES_manually_rejected/"
phase_dep_epoch_files = Path(phase_dep_epoch_dir).glob('*.fif*')
#evokeds_path = "/mnt/data/First_Project/AcuteNMES_EEG_Analysis/Epochs_NMES_manually_rejected/evokeds/"
evoked_GA_path = "/home/sara/NMES/analyzed data/Feb/evokeds/All_Conditions_Combined/"
evokeds_all_z = []
evokeds_all = []

for f in phase_dep_epoch_files:
    print('--- loading '+str(f)+' ---')
    epochs = mne.read_epochs(f, preload=True)
    #data_epoch_z = zscore(epochs._data, axis=1)
    #epochs_z = mne.EpochsArray(data_epoch_z, info = epochs.info,  tmin=epochs.tmin, events=epochs.events,
    #                               event_id = epochs.event_id, baseline =  epochs.baseline)
    #evokeds_z = epochs_z.average()
    evokeds = epochs.average()
    evoked_zscored = mne.baseline.rescale(evokeds._data, baseline=(-1, 1), times = evokeds.times , mode='zscore')
    evokeds_z = mne.EvokedArray(data = evoked_zscored, info = evokeds.info, tmin=evokeds.tmin, comment=evokeds.comment)
    evokeds_all.append(evokeds)
    evokeds_all_z.append(evokeds_z)



gfp = mne.grand_average(evokeds_all_z).data.std(axis=0, ddof=0)
   # Reproducing the MNE-Python plot style seen above
fig, ax = plt.subplots()
ax.plot(evokeds_all_z[0].times, gfp * 1e6, color='lime') 
    



Evoked_GrandAv = mne.grand_average(evokeds_all)
a  = Evoked_GrandAv.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
Evoked_GrandAv_z = mne.grand_average(evokeds_all_z)
Evoked_GrandAv_z.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
Evoked_GrandAv_z.plot(gfp=True, spatial_colors=True)

_,_, erp1_erp2 = get_ERP_1st_2nd(plot = True)
erp1_erp2.savefig('/home/sara/NMES/analyzed data/Feb/evokeds/All_Conditions_Combined/' + 'erp1_erp2')

mne.evoked.write_evokeds(evoked_GA_path + "Evokeds_all_conditions_ave.fif", Evoked_GrandAv_z) 
all_times = np.arange(-0.9, 0.6, 0.03)
top_ga = Evoked_GrandAv.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=10, nrows='auto')
top_ga.savefig('/home/sara/NMES/analyzed data/Feb/evokeds/All_Conditions_Combined/' + 'topo_ga')
erp_ga = Evoked_GrandAv.plot(spatial_colors = True, gfp = True) 
erp_ga.savefig('/home/sara/NMES/analyzed data/Feb/evokeds/All_Conditions_Combined/' + 'erp_ga')




