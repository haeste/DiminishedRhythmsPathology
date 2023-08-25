"""
@Mariellapanag

Created on Feb 2023
@author: mariella and chris, CNNP Lab

Reads in the timeseries of each bandpower for each patient and calculates the power of each biological rhythm.
output: saves these in the cycles folder, and produces 

"""

# external modules
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# internal modules
import proj_funcs.generic_funcs as generic_funcs
import proj_funcs.cycles_funcs as cycles_funcs


# %%
###### UPDATE this to where the data folder is stored ###############
in_dir = '../../../Data/reindexed_data/'

# list of patients

all_patients = [str(i) for i in range(0,39)]

n_patients = len ( all_patients )

extract_fluct_method = "bandpass"


# Choose variables to analyse
variables_analyse = "rel_log_roi_bp"
srate = 24 * 60 * 2

# folder for fixed frequencies analysis

## cycles periods info
fluct_name = ["20min-40min", "40min-1h", "1h-3h", "3h-6h", "6h-9h", "9h-12h", "12h-19h", "19h-1.3d", "1.3d-2d" ,"2d-3d"]

# frequency ranges
fluct_narrowband = {"20min-40min": [35,71], "40min-1h":[23,35], "1h-3h": [8, 23], "3h-6h": [4, 8], "6h-9h": [2.6, 4],
                    "9h-12h": [2, 2.6], "12h-19h": [1.3, 2], "19h-1.3d": [0.8, 1.3], "1.3d-2d":[0.5, 0.8], "2d-3d":[0.33,0.5]}

# ## For every subject save data and make plots
for i in np.arange ( 0,n_patients):  # range(n_patients):
    my_patient = all_patients[i]

    print ( my_patient )
    # input path for data

    # output files
    output_dir = os.path.join ( in_dir, "cycles")
    os.makedirs ( output_dir, exist_ok=True )


    # load
    # Input path of data transformed
    fname = os.path.join ( in_dir,'bandpower_series',
                           "{}_rel_log_roi_bp_scale60.mat".format ( my_patient) )
    
    rel_log_bp_dict = {}
    generic_funcs.mat_load ( fname, rel_log_bp_dict,variables_analyse, "freq_band", "roi_names", "t_days", "time_of_day",
                             "n_roi", "n_win")

    rel_log_bp_dict['vars_analyse'] = rel_log_bp_dict[variables_analyse]
    rel_log_bp_dict['freq_band'] = np.array ( [a.replace ( " ", "" ) for a in rel_log_bp_dict['freq_band'].squeeze ()] )
    rel_log_bp_dict['roi_names'] = np.array ( [a.replace ( " ", "" ) for a in rel_log_bp_dict['roi_names'].squeeze ()] )

    rel_log_bp_dict['t_days'] = rel_log_bp_dict['t_days'].squeeze()
    rel_log_bp_dict['time_of_day'] = np.concatenate(rel_log_bp_dict['time_of_day'].squeeze())

    rel_log_bp_dict['n_win'] = int ( rel_log_bp_dict['n_win'].flatten () )
    rel_log_bp_dict['n_roi'] = int ( rel_log_bp_dict['n_roi'].flatten () )
    
    rel_log_bp_dict['roi_names'] = np.array ( [a.replace ( " ", "" ) for a in rel_log_bp_dict['roi_names'].squeeze ()] )

    rel_log_bp_dict['fluct_name'] = fluct_name
    rel_log_bp_dict['fluct_narrowband'] = fluct_narrowband
    rel_log_bp_dict['srate'] = srate
    rel_log_bp_dict['extract_fluct_method'] = extract_fluct_method
    # Extract cycles for each variable of interest
    # using band-pass filter
    for index, val in enumerate(rel_log_bp_dict['freq_band'][:-1]):

        [cycles_output, cycles_output_imp, roi_data_imp, roi_data_imp_tmp] = cycles_funcs.get_cycles_bandpass ( rel_log_bp_dict['vars_analyse'][:,:, index], fluct_name, fluct_narrowband, srate )
        rel_log_bp_dict['cycles_output'] = cycles_output
        rel_log_bp_dict['cycles_output_imp'] = cycles_output_imp
        rel_log_bp_dict['val'] = val
        cycles_fpath = os.path.join ( output_dir, "{}_ROI_{}_CYCLES_scale60_{}.mat".format ( my_patient, variables_analyse, val))
        cycles_all = generic_funcs.mat_save(cycles_fpath , rel_log_bp_dict, "cycles_output", "cycles_output_imp", 'val', "srate", "fluct_name", "fluct_narrowband",
                             "extract_fluct_method", "n_win", "n_roi", "roi_names", "t_days", "time_of_day")

    del cycles_all
del i


