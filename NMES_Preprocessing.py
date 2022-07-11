

#%%


import mne
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from autoreject import AutoReject
from mne.channels import make_standard_montage
import NMES_Preprocessing_functions as prepross

#%%


exdir = "/home/sara/NMES/NMES_Experimnet/LaCa/"
files = list(Path(exdir).glob('**/*.xdf'))


lists_bad_channels = []
lists_pluses = []
for f in files:
    plt.close('all')
    
    # Step 1 : reading XDF files and changing their format to raw mne 
    raw, pulses, pulses_corrected = prepross.XDF_correct_time_stamp_reject_pulses(f)
    lists_pluses.append([str(f.parts[5]) +  str(f.parts[-1][39:43]), pulses, pulses_corrected])
    #raw.resample(250, npad="auto")  # set sampling frequency to 500Hz or maybe 250Hz because my PC crashes all the time

    # Preprocessing EEG channels 
             
    # Selecting EEG channels
    raw_eeg = raw.drop_channels(['EDC_L', 'EDC_R', 'ECR_L', 'ECR_R', 'FCR_L', 'FCR_R', 'FDS_L', 'FDS_R', 'EKG'])
    print('Number of channels in raw_eeg:')
    print(len(raw_eeg.ch_names), end=' → drop nine → ')
    
    
    
    # step 2: apply regression for ocular artifacts
    print('starting regress out eye artifacts')
    raw_regr = prepross.regress_out_pupils(raw_eeg)
    raw_regr.notch_filter((50,100))
    #raw_regr.plot(n_channels=64)
    
    
    # step 3 : Rejecting bad channels based on the visualization of variance of channels
    # plotting of channel variance

    eeg_regr_interp, badchans_threshold  = prepross.mark_bad_channels_interpolate(f.parts,raw_regr)
    # Creating a list of subjects and bad channels that were rejected
    lists_bad_channels.append([str(f.parts[5]) + str(f.parts[-1][39:43]), badchans_threshold])
    
  
    # Step 4 : Creating epochs 
    
    # 4.1. Create events from the annotations present in the raw file
    # excluding non-unique events and time-stamps
    (events_from_annot, event_dict) = mne.events_from_annotations(eeg_regr_interp)
    
    u, indices = np.unique(events_from_annot[:,0], return_index=True)
    events_from_annot_unique = events_from_annot[indices]
    event_unique, event_unique_ind  = np.unique(events_from_annot_unique[:,2], return_index=True)
    
    # 4.2. Create epochs based on the events, from -1 to 1s
    # Set the baseline to None and None, because mne suggests to do a baseline correction after ICA
    epochs = mne.Epochs(raw_regr, events_from_annot_unique, event_id=event_dict,
                        tmin=-1, tmax=1, reject=None, preload=True,  baseline=(0, 0))
    
    
    # 4.3. filtering, since we want to see the effect of 40 hz target frequency, the low pass was selected on 45
    epochs.filter(0.5, 45, method='iir', verbose=0)

    
    # 4.4.  Applying ICA after filtering and before baseline correction 
    data_ica  = prepross.clean_dataset(epochs)


    # 4.5. Applying baseline, this is based on what teasa toolbox suggested  
    epochs = data_ica['eeg'].apply_baseline(baseline=(-0.9, -0.1))    
    montage = make_standard_montage('standard_1005')
    epochs = epochs.set_montage(montage)

    # 4.6. Removing bad trials using autoreject https://autoreject.github.io/stable/index.html
    ar = AutoReject(n_interpolate=[0], 
                    consensus=[0.3], 
                    n_jobs=-1)

    clean_epochs = ar.fit(epochs).transform(epochs)   
    evokeds = clean_epochs.average()
    all_times = np.arange(-0.8, 0.8, 0.1)
    fig_topo = evokeds.plot_topomap(all_times, ch_type='eeg', time_unit='s', ncols=8, nrows='auto')
    fig_erp = evokeds.plot(spatial_colors = True, gfp=True)

    

    
    # Save epoch files and figures
    save_folder = "/home/sara/NMES/analyzed data/Feb/epochs/"
    save_folder_figs = "/home/sara/NMES/analyzed data/Feb/epochs/figs/"
    clean_epochs.save(save_folder+ str(f.parts[5]) + '_' + str(f.parts[-1][39:43]) + '_epo.fif', overwrite = True, split_size='2GB')
    fig_topo.savefig(save_folder_figs + 'topo' + '_' + str(f.parts[5]) + '_' + str(f.parts[-1][39:43]) + '_' + '.png')
    fig_erp.savefig(save_folder_figs + 'erp' + '_' + str(f.parts[5]) + '_' + str(f.parts[-1][39:43]) + '_' + '.png')
    

cols_bad_chans = ['Subject', 'Bad Channels']
df_bad_chans = pd.DataFrame(lists_bad_channels, columns = cols_bad_chans)   
df_bad_chans.to_csv (save_folder +'Subject_ channel_rejected.csv', index = None, header=True)   

cols = ['','', 'Pulses', 'Pulses Corrected']
lists_pluses.append(['', '',  pulses , pulses_corrected])
df_pulses = pd.DataFrame(lists_pluses, columns=cols)     
df_pulses.to_csv (save_folder +'Subject_ pulse.csv', index = None, header=True) 
    
    
    