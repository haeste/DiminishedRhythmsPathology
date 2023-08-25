"""
@Mariellapanag

Created on May 2023
@author: mariella and chris, CNNP Lab
Calculates the AUC of the cycles power on SOZ against other regions. 
output: saves these to the AUCs folder, and produces figures 1 (a,d) and 2 (a,c)

"""

# external modules
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
# internal modules
import proj_funcs.generic_funcs as generic
import proj_funcs.cycles_funcs as cycles_funcs
import proj_funcs.funcs_Drs as funcs_Drs
import proj_funcs.vis_cycles as vis_cycles
import proj_funcs.vis_funcs as vis_funcs

# %%
###### UPDATE this to where the data folder is stored ###############
in_dir = '../../../Data/reindexed_data/'


all_patients = [str(i) for i in range(0,39)]

n_patients = len ( all_patients )

extract_fluct_method = "bandpass"


with open(in_dir + 'SOZ_info.pickle', 'rb') as handle:
    SOZ_info = pickle.load(handle)
# Choose variables to analyse
variables_analyse = "rel_log_roi_bp"

# frequency bands to look
fb_interest = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
AUC_info_allP = []
# ## For every subject save data and make plots
for i in np.arange ( 0, n_patients ):  # range(n_patients):
    my_patient = all_patients[i]
    
    print ( my_patient )

    out_dir = os.path.join ( in_dir, "AUCs" )
    os.makedirs ( out_dir, exist_ok=True )
    plot_dir = os.path.join ( in_dir, "plots", 'figure1and2a' )
    os.makedirs ( plot_dir, exist_ok=True )

    # load information about SOZ
    # output files
    # get SOZ info
    soz_info_tmp = SOZ_info[my_patient]

    # load data
    for var in fb_interest:
        # input files
        bandpower_dir = os.path.join ( in_dir, "bandpower_series")

        fname = os.path.join ( in_dir, 'cycles', "{}_ROI_{}_CYCLES_scale60_{}.mat".format ( my_patient, variables_analyse, var) )
        cycles_dict = {}
        generic.mat_load ( fname, cycles_dict, "cycles_output", "val", "srate", "fluct_name", "fluct_narrowband",
                           "extract_fluct_method", "n_roi", "roi_names",
                           "t_days", "time_of_day")
        val = cycles_dict['val'][0]
        # check whether we are analysing the correct data
        if var == val:
            print("Correct data loaded")
        else:
            print("Error: Data loaded are not the right ones")

        if cycles_dict['fluct_name'].shape[0] != 1:
            fluct_name = np.array([a.replace(" ", "") for a in cycles_dict['fluct_name'].squeeze()])
        else:
            fluct_name = cycles_dict['fluct_name']
        roi_names = np.array([a.replace(" ", "") for a in cycles_dict['roi_names'].squeeze()])

        n_roi = int ( cycles_dict['n_roi'].flatten () )

        # combine the SOZ with the info about roi names for subject
        roi_names_df = pd.DataFrame ( {"roi_names": roi_names} )
        # merge dataframes
        soz_df_final = pd.merge ( roi_names_df, soz_info_tmp, how="left", on=["roi_names"] )

        #
        is_soz = soz_df_final["is_soz"].to_numpy()

        t_days = cycles_dict['t_days'].squeeze ()
        ########### start computing measures and Drs across all cycles
        n_cycles = len ( fluct_name )

        print("compute AUC for {}".format(var))
        cycles_allmeasures = np.zeros ( (n_roi, n_cycles) )

        cycles_output = cycles_dict['cycles_output']

        ## Extract the measures of interest for these cycles
        # compute power of cyclic fluctuation
        cycles_power = cycles_funcs.power_signal(cycles_dict['cycles_output'])

        ## Compute the Drs measure between the SOZ and non-SOZ
        auc_variable = funcs_Drs.calculate_DRS_power ( cycles_power, is_soz )
        MINUTES_PER_DAY = 1440
        hour_of_day = [(pd.Timestamp(x).hour*60 + pd.Timestamp(x).minute)  for x in cycles_dict['time_of_day'].astype('datetime64[ns]')]
        hour_of_day_2 = [x/MINUTES_PER_DAY if x>0 else 0 for x in hour_of_day]
        t_arr = cycles_dict['time_of_day'].astype('datetime64[ns]')
        t_days_time = t_days+hour_of_day_2[0]

        ## input path of variables
        filename = os.path.join ( out_dir, '{}_ROI_AUC_scale60_SOZ_{}.mat'.format(my_patient, val) )

        cycles_dict['cycles_power'] = cycles_power
        cycles_dict['auc_variable'] = auc_variable
        cycles_dict['val'] = val
        cycles_dict['fluct_name'] = fluct_name
        cycles_dict['is_soz'] = is_soz
        cycles_dict['roi_names'] = roi_names
        cycles_dict['n_cycles'] = n_cycles
        cycles_dict['n_roi'] = n_roi
        generic.mat_save (filename, cycles_dict, 'cycles_power', 'auc_variable', "val",
                           "fluct_name", 'is_soz',
                           'roi_names', 'n_cycles', "n_roi")
        
        # compute the median of the average power of each channel within each cycle
        power_median = np.median(cycles_power, 0)
        # compute the median of the average power of each channel for channels in non-SOZ only within each cycle
        id_rows_nonSOZ = is_soz == 0
        power_median_nonSOZ = np.median(cycles_power[id_rows_nonSOZ, :], 0)

        # compute the median of the average power of each channel for channels in SOZ only within each cycle
        id_rows_SOZ = is_soz == 1
        power_median_SOZ = np.median ( cycles_power[id_rows_SOZ, :], 0 )

        # dataframe with info for each patient
        auc_pat_df = pd.DataFrame({"AUC": auc_variable, "power_median": power_median,
                                   "power_median_SOZ": power_median_SOZ,
                                   "power_median_nonSOZ": power_median_nonSOZ,
                                   "subject": np.repeat(my_patient, len(fluct_name)),
                                     "fluct_name": fluct_name,
                                     'band':var})
        rows_df = auc_pat_df.shape[0]
        AUC_info_allP.append(auc_pat_df)

        if my_patient == "0":
            ##################  make plots for cycles figures 1(a) and 2(a) ####################################
            for cc in [3,7]:
    
                # select one cycle at a time
                var_1cycle = cycles_dict['cycles_output'][:,:, cc]
    
                # set ylim for all plots (to be able to compare different cycles across channels)
                ymin = np.nanmin ( var_1cycle )
                ymax = np.nanmax ( var_1cycle )
                cm = 1/2.54 
                fig, ax = plt.subplots ( n_roi, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(13*cm, 9
                                                                                                  *cm) )
                axs1 = ax[:, 0]
                fig_title = "Subject {} \n {} \n Cycle period: {}".format(my_patient, var, fluct_name[cc])
                vis_cycles.plot_cycles_all_t_days ( t_days_time, roi_names=roi_names, roi_is_resect=is_soz,
                                         cycle_channels=var_1cycle, display_yticks=False, label_coords=None,
                                         fig_title=fig_title, ax=axs1, ylim=[ymin, ymax], outcome_clrs=None )
                
                TEN_THRITY = 22.5/24
                SEVEN_THIRTY = 7.5/24
                for ax_i in axs1:
                    ax_i.fill_between(t_days_time, *ax_i.get_ylim(), where=((t_days_time%1) > TEN_THRITY) | ((t_days_time%1) < SEVEN_THIRTY) , facecolor='k', alpha=.1)
    
                # power
                cycles_power_1cycle = cycles_power[:, cc]
                drs_1cycle = auc_variable[cc]
    
                # power barplot
                axs2 = ax[:, 1]
                xmax = np.nanmax ( cycles_power_1cycle ) + np.std ( cycles_power_1cycle )
                vis_cycles.plot_barplot_measure ( cycles_power_1cycle, is_soz, xlim=[0, xmax], fig_title="power (a.u) \n AUC: {}".format(round(drs_1cycle, 3)), ax=axs2,
                               outcome_clrs=None )
                
                fig.subplots_adjust ( left=0.2, bottom=0.08, right=0.97, top=0.99, wspace=0.05, hspace=0 )
                fname = "{}_Cycles_and_measures_SOZ_{}_{}".format ( my_patient, var, fluct_name[cc] )
                generic.save_plot ( plot_dir, fname, plot_formats=["pdf"] )
                plt.close ( fig )

auc_comb_allP = pd.concat ( AUC_info_allP )
auc_comb_allP.to_csv (
    os.path.join ( out_dir, "AUCs.csv".format ( var ) ) )
#%% Figures 1 (d) and 2 (c)
fluct_name = fluct_name[2:7]

auc_comb_allP['fluct_name'] = auc_comb_allP.fluct_name.str.strip()
drs_df_delta = auc_comb_allP[auc_comb_allP.band=='Delta'].reset_index()

fig, ax = plt.subplots ( 1, 1, figsize=(2.5, 3.5) )
fig.tight_layout ( rect=[0.14, 0.14, 0.87, 0.85], h_pad=3, w_pad=3 )
drs_df_delta['drs'] = drs_df_delta.AUC
vis_funcs.plot_AUC_per_cycle(drs_df_delta[drs_df_delta.fluct_name=='19h-1.3d'],x_var='fluct_name', ax = ax, size=5)
plt.ylim([0,1])
plt.show()

fig, ax = plt.subplots ( 1, 1, figsize=(5, 3) )
fig.tight_layout ( rect=[0.14, 0.14, 0.87, 0.85], h_pad=3, w_pad=3 )
fluct_name
for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)

vis_funcs.plot_AUC_per_cycle(drs_df_delta.loc[(drs_df_delta.fluct_name!='19h-1.3d') & (drs_df_delta.fluct_name.isin(fluct_name))],x_var='fluct_name', ax=ax, overlay_violin=True, size=5)
plt.ylim([0,1])

plt.show()





