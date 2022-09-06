 Acute NMES Study
 Order of scripts:
 1. NMES_Preprocessing.py reads XDF files and, after preprocessing using NMES_Preprocessing_functions, saves epoch files. 
 2. Bad channels were identified by visual inspection using Channel_interpolation_manually.py 
 3. Right now, we are focusing on real-time labels. Diagonal_shift_GA_mod_just_real_time.py and diagonal_shift_each_sub_real_time.py are the two scripts
    for phase analysis, respectively, for group avg and individual subjects. 
 4. peak_vs_trough_new.py script uses real-time labels to make a cluster-based statistical comparison between the negative and positive peak conditions.
