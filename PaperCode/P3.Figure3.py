#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:24:03 2023

@author: nct76
"""

"""
@Mariellapanag

Created on March 2023
@author: mariella and chris
Calculates a mixed effects model to show that SOZ has an effect on cycle power above that of a region specific effect. 
Produces figures 3 (a and b)

"""

# external modules
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as io
import pandas as pd
# internal modules
import proj_funcs.generic_funcs as generic_funcs
import proj_funcs.vis_funcs as vis_funcs


#%%
###### UPDATE this to where the data folder is stored ###############
in_dir = '../../../Data/reindexed_data/'
all_patients = [str(i) for i in range(0,39)]

n_patients = len ( all_patients )

# exclude subject 21 from the list of patients as it has not clear SOZ
all_patients = [pat for pat in all_patients if pat != "21"]

n_patients = len ( all_patients )

# Choose variables to analyse
variables_analyse = "rel_log_roi_bp"

# folder for fixed frequencies analysis
var_folder = "rel_log_roi_bp_FIXED_freqs"

# frequency bands to look
fb_interest = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

out_subfolder = "Cycle_Pow_ROI"

# plot
plot_type = 'beeswarm'
roi_grouping = {'l.inferiorparietal_1':'Parietal', 'l.isthmuscingulate_1':'Cingulate',
       'l.lateraloccipital_1':'Occipital', 'l.lingual_1':'Occipital', 'l.pericalcarine_1':'Occipital',
       'l.precuneus_1':'Parietal', 'l.inferiorparietal_2':'Parietal', 'l.lateraloccipital_2':'Occipital',
       'l.precuneus_2':'Parietal', 'l.superiorparietal_2':'Parietal', 'l.superiorparietal_3':'Parietal',
       'Left-Hippocampus':'Hippocampus', 'l.bankssts_1':'Temporal', 'l.cuneus_1':'Occipital',
       'l.inferiortemporal_1':'Temporal', 'l.middletemporal_1':'Temporal',
       'l.parahippocampal_1':'Temporal', 'l.parsopercularis_1':'Frontal',
       'l.rostralmiddlefrontal_1':'Frontal', 'l.superiorparietal_1':'Parietal',
       'l.superiortemporal_1':'Temporal', 'l.supramarginal_1':'Parietal', 'l.lingual_2':'Occipital',
       'l.middletemporal_2':'Temporal', 'l.postcentral_2':'Parietal', 'l.postcentral_3':'Parietal',
       'l.precentral_3':'Frontal', 'l.precentral_4':'Frontal', 'l.supramarginal_2':'Parietal',
       'Left-Amygdala':'Amygdala', 'l.entorhinal_1':'Temporal', 'l.lateralorbitofrontal_1':'Frontal',
       'l.medialorbitofrontal_1':'Frontal', 'l.parsorbitalis_1':'Frontal',
       'l.superiorfrontal_1':'Frontal', 'l.frontalpole_1':'Frontal', 'l.transversetemporal_1':'Temporal',
       'l.insula_1':'Temporal', 'l.fusiform_2':'Temporal', 'l.lateralorbitofrontal_2':'Frontal',
       'l.rostralmiddlefrontal_2':'Frontal', 'l.rostralmiddlefrontal_3':'Frontal',
       'l.superiortemporal_2':'Temporal', 'l.insula_2':'Temporal', 'Right-Hippocampus':'Hippocampus',
       'l.inferiortemporal_2':'Temporal', 'r.caudalmiddlefrontal_1':'Frontal',
       'r.inferiortemporal_1':'Temporal', 'r.middletemporal_1':'Temporal', 'r.precentral_1':'Frontal',
       'r.rostralmiddlefrontal_1':'Frontal', 'r.superiorfrontal_1':'Frontal',
       'r.superiortemporal_1':'Temporal', 'r.supramarginal_1':'Parietal', 'r.middletemporal_2':'Temporal',
       'r.postcentral_2':'Parietal', 'r.rostralmiddlefrontal_2':'Frontal',
       'l.caudalmiddlefrontal_1':'Frontal', 'l.postcentral_1':'Parietal', 'l.precentral_2':'Frontal',
       'l.superiorfrontal_3':'Frontal', 'l.superiorfrontal_4':'Frontal',
       'r.parsopercularis_1':'Frontal', 'r.parstriangularis_1':'Frontal', 'r.postcentral_1':'Parietal',
       'r.transversetemporal_1':'Temporal', 'r.insula_1':'Temporal', 'r.precentral_2':'Frontal',
       'r.precentral_3':'Frontal', 'r.superiorfrontal_2':'Frontal', 'r.superiortemporal_2':'Temporal',
       'r.supramarginal_2':'Parietal', 'l.parstriangularis_1':'Frontal', 'l.precentral_1':'Frontal',
       'Left-Caudate':'Caudate', 'Left-Putamen':'Putamen', 'l.rostralanteriorcingulate_1':'Cingulate',
       'l.temporalpole_1':'Temporal', 'l.fusiform_1':'Temporal', 'Right-Amygdala':'Amygdala',
       'r.bankssts_1':'Temporal', 'r.fusiform_1':'Temporal', 'r.inferiorparietal_1':'Parietal',
       'r.lateraloccipital_1':'Occipital', 'r.lingual_1':'Occipital', 'r.inferiorparietal_2':'Parietal',
       'r.inferiorparietal_3':'Parietal', 'r.inferiortemporal_2':'Parietal',
       'r.lateraloccipital_2':'Occipital', 'r.lateraloccipital_3':'Occipital',
       'r.superiorparietal_2':'Parietal', 'r.entorhinal_1':'Temporal', 'r.isthmuscingulate_1':'Cingulate',
       'r.parahippocampal_1':'Temporal', 'r.temporalpole_1':'Temporal', 'r.fusiform_2':'Temporal',
       'Right-Caudate':'Caudate', 'Right-Putamen':'Putamen', 'r.insula_2':'Temporal',
       'l.caudalanteriorcingulate_1':'Cingulate', 'l.superiorfrontal_2':'Frontal',
       'r.precuneus_1':'Parietal', 'r.superiorparietal_1':'Parietal', 'r.superiorparietal_3':'Parietal',
       'r.caudalanteriorcingulate_1':'Cingulate', 'r.paracentral_1':'Frontal',
       'r.posteriorcingulate_1':'Cingulate', 'r.precuneus_2':'Parietal', 'r.superiorfrontal_3':'Frontal',
       'r.superiorfrontal_4':'Frontal', 'r.lateralorbitofrontal_1':'Frontal',
       'r.medialorbitofrontal_1':'Frontal', 'r.lateralorbitofrontal_2':'Frontal',
       'r.medialorbitofrontal_2':'Frontal', 'l.paracentral_1':'Frontal',
       'Left-Thalamus-Proper':'Thalamus', 'Right-Thalamus-Proper':'Thalamus',
       'r.parsorbitalis_1':'Frontal', 'r.rostralanteriorcingulate_1':'Cingulate',
       'l.posteriorcingulate_1':'Cingulate'}
tmp_plots = True
cycles_data_all = []
cycles_all_df_long = pd.DataFrame()
DRS_vals = {}
drs_df = pd.DataFrame()
for i in np.arange ( 0, n_patients ):  # range(n_patients):
    my_patient = all_patients[i]

    print(f'Patient {my_patient}')

    ## Load the dictionary
    n_roi = None
    for var in fb_interest:

        # the measures and drs data
        f_meas_drs_path = os.path.join ( in_dir,'AUCs',
                                         '{}_ROI_AUC_scale60_SOZ_{}.mat'.format ( my_patient, var ) )
        cycles_data = io.loadmat(f_meas_drs_path)
        cycles_data_all.append(cycles_data)
        drs = {}
        for cyc_i in range(0,cycles_data['n_cycles'].flatten()[0]):
            drs[str.strip(cycles_data['fluct_name'].flatten()[cyc_i])] = cycles_data['auc_variable'].flatten()[cyc_i]
            drs['EEG_band'] = var
            drs['patient'] = my_patient
        DRS_vals[var] = drs
        drs  = {k:[v] for k,v in drs.items()}
        drs_df = pd.concat([drs_df, pd.DataFrame(drs)])
        for roi in range(0,cycles_data['n_roi'].flatten()[0]):
            for cyc_i in range(0,cycles_data['n_cycles'].flatten()[0]):
                cycles_data_r = cycles_data.copy()
                cycles_data_r['EEG_band'] = var
                cycles_data_r['cycles_power'] = cycles_data['cycles_power'][roi,cyc_i]
                
                s = str.strip(cycles_data['roi_names'][roi])
                cycles_data_r['lobe'] = roi_grouping[s]
                
                cycles_data_r['roi_names']  = s
                cycles_data_r['is_soz'] = cycles_data['is_soz'].flatten()[roi]
                cycles_data_r['fluct_name'] = cycles_data['fluct_name'][cyc_i]
                cycles_data_r['patient'] = my_patient
                cycles_data_r = {k:[cycles_data_r[k]] for k in ('patient','EEG_band','roi_names','fluct_name','cycles_power','is_soz','temporal','lobe',) if k in cycles_data_r}
                cycles_all_df_long = pd.concat([cycles_all_df_long, pd.DataFrame(cycles_data_r)])

#%%
roi_order = ['Hippocampus','parahippocampal','entorhinal','inferiortemporal',
             'middletemporal','superiortemporal','transversetemporal','temporalpole',
             'inferiorparietal', 'isthmuscingulate', 'lateraloccipital',
             'lingual', 'pericalcarine', 'precuneus', 'superiorparietal',
             'bankssts', 'cuneus', 'parsopercularis','rostralmiddlefrontal', 
             'supramarginal', 'postcentral', 'precentral', 'Amygdala', 
             'lateralorbitofrontal', 'medialorbitofrontal', 'parsorbitalis',
             'superiorfrontal', 'frontalpole',  'insula',
             'fusiform', 'caudalmiddlefrontal', 'parstriangularis', 'Caudate',
             'Putamen', 'rostralanteriorcingulate', 
             'caudalanteriorcingulate', 'paracentral', 'posteriorcingulate',
             'Thalamus-Proper']
lobe_order = ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Cingulate', 'Hippocampus', 'Amygdala']
median_drs = drs_df.groupby(['EEG_band']).median()
import seaborn as sns

#%%
fig, ax = plt.subplots ( 1, 1, figsize=(7.5, 4.5) )
fig.tight_layout ( rect=[0.14, 0.14, 0.87, 0.85], h_pad=0, w_pad=0 )
#sns.set(rc={'figure.figsize':(16,6)})
sns.set_style("ticks")
for fb in ['Delta']:
    for fluctName in [cycles_all_df_long.fluct_name.unique()[3]]:
        #plt.figure()
        df_long_delta = cycles_all_df_long[cycles_all_df_long.EEG_band==fb]
        df_long_delta_c = df_long_delta[df_long_delta.fluct_name==fluctName]
        df_long_delta_c['cycles_power'] = df_long_delta_c.cycles_power*1000
        df_long_delta_c = df_long_delta_c[~df_long_delta_c['lobe'].isin(['Caudate', 'Putamen', 'Thalamus'])]
        df_long_delta_c['roi_names'] = df_long_delta_c.roi_names.astype('category')
        df_long_delta_c['lobe'] = df_long_delta_c.lobe.astype('category')
        df_long_delta_c['patient'] = df_long_delta_c.patient.astype('category')
        df_long_delta_c_g = df_long_delta_c.groupby(['patient','lobe']).agg({'EEG_band':'first','fluct_name':'first', 'cycles_power':np.median,'is_soz':np.median}).reset_index()
        df_long_delta_c_g_soz = df_long_delta_c.groupby(['patient','lobe','is_soz']).agg({'EEG_band':'first','fluct_name':'first', 'cycles_power':np.median}).reset_index()
        palette = [sns.color_palette("Blues")[-2], sns.color_palette("Reds")[-2]]
        ax = sns.stripplot(data=df_long_delta_c_g_soz,x='lobe',y='cycles_power', hue='is_soz',order=lobe_order, ax=ax,palette=palette)
        median_width = 0.4
        for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
            sample_name = text.get_text()  # "X" or "Y"
    
            # calculate the median value for all replicates of either X or Y
            median_val = df_long_delta_c_g[df_long_delta_c_g['lobe']==sample_name].cycles_power.median()
            median_val_soz =df_long_delta_c_g_soz[df_long_delta_c_g_soz['lobe']==sample_name].cycles_power.median()
            soz_median = df_long_delta_c_g_soz[(df_long_delta_c_g_soz['lobe']==sample_name) & (df_long_delta_c_g_soz['is_soz']==1)].cycles_power.median()
            other_median = df_long_delta_c_g_soz[(df_long_delta_c_g_soz['lobe']==sample_name) & (df_long_delta_c_g_soz['is_soz']==0)].cycles_power.median()
            soz_quantiles = df_long_delta_c_g_soz[(df_long_delta_c_g_soz['lobe']==sample_name) & (df_long_delta_c_g_soz['is_soz']==1)].cycles_power.quantile([0.25, 0.75])
            other_quantiles = df_long_delta_c_g_soz[(df_long_delta_c_g_soz['lobe']==sample_name) & (df_long_delta_c_g_soz['is_soz']==0)].cycles_power.quantile([0.25, 0.75])
            soz_iqr = soz_quantiles[0.75] - soz_quantiles[0.25]
            other_iqr = other_quantiles[0.75] - other_quantiles[0.25]
            # ax.plot([tick-median_width/2, tick+median_width/2], [median_val_soz,median_val_soz] ,
            #              lw=4, color='k', marker='_')
            ax.plot([tick+median_width/2], [soz_median] ,linestyle='-', color='r')
            ax.plot([tick+median_width/2+0.1], [other_median],linestyle='-', color='b')
            ax.vlines( [tick+median_width/2], soz_quantiles[0.25], soz_quantiles[0.75], color='r')
            ax.vlines( [tick+median_width/2+0.1], other_quantiles[0.25], other_quantiles[0.75], color='b')
            ax.plot([tick+median_width/2], [soz_median] ,
                          lw=1, color='r', marker='_')
            ax.plot([tick+median_width/2+0.1], [other_median] ,
                          lw=1, color='b', marker='_')
        plt.ylabel(fluctName)
        plt.xlabel('')
        plt.xticks(rotation=45, ha='right')
        drs_of = median_drs[str.strip(fluctName)][fb]
        plt.title(fb + ' (AUC=' + str(round(drs_of,2)) +')')
        # plt.savefig(os.path.join ( plot_dir_all, '{}_Cycles_{}_POW_ROI_SOZ.pdf'.format(str.strip(fluctName),fb) ), bbox_inches='tight')
        # plt.close('all')
        
plt.show()
#%%

#%%
import warnings
warnings.filterwarnings("ignore")
fluctNames = cycles_all_df_long.fluct_name.unique()[2:-2]
cycle_band_mat_p_LR = np.array([[np.nan]*len(fluctNames)]*len(fb_interest))
cycle_band_mat_coef = np.array([[np.nan]*len(fluctNames)]*len(fb_interest))
cycle_band_mat_pred = np.array([[np.nan]*len(fluctNames)]*len(fb_interest))
cycle_band_mat_conf = np.array([[(np.nan, np.nan)]*len(fluctNames)]*len(fb_interest))
cycle_band_mat_p = np.array([[np.nan]*len(fluctNames)]*len(fb_interest))
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
for v_i, var in enumerate(fb_interest):
    print(var)
    df_long_fb= cycles_all_df_long[cycles_all_df_long.EEG_band==var]
    for c_i, cyc_name in enumerate(fluctNames):
        # if cyc_name == '19h-1.3d' and var=='Delta':
        #     continue
        print(cyc_name)
        df_long_fb2 = df_long_fb[df_long_fb.fluct_name==cyc_name]
        df_long_fb2.drop_duplicates(inplace=True)
        df_long_fb2['cycles_power_ss'] = (df_long_fb2['cycles_power']- df_long_fb2['cycles_power'].min()) / (df_long_fb2['cycles_power'].max() - df_long_fb2['cycles_power'].min())
       
        df_z = pd.get_dummies(df_long_fb2[['cycles_power', 'lobe', 'is_soz']], columns=['lobe']).astype(float)
        df_z['patient'] = df_long_fb2['patient']
        df_z['cycles_power'] = df_z[['cycles_power']].apply(stats.zscore)
        
        mdempty = smf.mixedlm("cycles_power ~ 1", df_z, groups=df_z['patient'])
        mdfempty  = mdempty.fit(reml=False)
        
        md = smf.mixedlm("cycles_power ~ lobe_Amygdala + lobe_Caudate + lobe_Cingulate + lobe_Frontal + lobe_Hippocampus + lobe_Occipital + lobe_Parietal + lobe_Putamen + lobe_Temporal + lobe_Thalamus + is_soz", df_z, groups=df_z['patient'])
        mdfnested = md.fit(reml=False)
        cycle_band_mat_p[v_i,c_i] = mdfnested.pvalues['is_soz']
        nestedllf = mdfnested.llf
        md = smf.mixedlm("cycles_power ~ lobe_Amygdala + lobe_Caudate + lobe_Cingulate + lobe_Frontal + lobe_Hippocampus + lobe_Occipital + lobe_Parietal + lobe_Putamen + lobe_Temporal + lobe_Thalamus" , df_z, groups=df_z['patient'])
        mdfred = md.fit(reml=False)
        redllf = mdfred.llf
        
        
        LR_statistic = -2*(redllf-nestedllf)
        p_val = stats.chi2.sf(LR_statistic, 2)
        print('llfred: ' + str(redllf) + ' llfnested: ' + str(nestedllf))
        print('LR stat: ' + str(LR_statistic) + ' pval: ' + str(round(mdfnested.pvalues['is_soz'], 3)) 
              + '(' + str(round(p_val,3)) + ')')
        cycle_band_mat_p_LR[v_i,c_i] = p_val
        temp = pd.DataFrame({'lobe_Amygdala':[0] ,'lobe_Caudate':[0],  'lobe_Cingulate':[0], 'lobe_Frontal':[0], 'lobe_Hippocampus':[0], 'lobe_Occipital':[0], 'lobe_Parietal':[0], 'lobe_Putamen':[0], 'lobe_Temporal':[1], 'lobe_Thalamus':[0], 'is_soz':[1]})
        
        cycle_band_mat_coef[v_i,c_i] = mdfnested.params['is_soz']
        conf_ints = mdfnested.conf_int()
        cycle_band_mat_conf[v_i,c_i] = conf_ints[conf_ints.index=='is_soz']


#%%
sns.set_style('ticks')
#plt.bar([0,1,2,3,4,5], cycle_band_mat_coef[0,:])
colours = sns.color_palette("tab10",7)
colours = colours[1:]
colours = colours[0:2] + colours[3:6]


for v_i, var in enumerate(fb_interest):
    for cyc in [0,1,2,3,4,5]:
        if cyc==0:
            plt.plot( cyc+(v_i/10),  cycle_band_mat_coef[v_i,cyc], color=colours[v_i], marker='_', label=var)
            plt.plot([cyc+(v_i/10),cyc+(v_i/10)] ,  cycle_band_mat_conf[v_i,cyc], color=colours[v_i], marker='_')
        else:
            plt.plot( cyc+(v_i/10),  cycle_band_mat_coef[v_i,cyc], color=colours[v_i], marker='_')
            plt.plot([cyc+(v_i/10),cyc+(v_i/10)] ,  cycle_band_mat_conf[v_i,cyc], color=colours[v_i], marker='_')
plt.xticks([0,1,2,3,4,5], labels= ['1h-3h', '3h-6h', '6h-9h', '9h-12h','12h-19h', '19h-31h'])
plt.plot([0,6],[0,0], 'k--')
plt.ylim([-.45, .45])
plt.xlabel('Cycle')
plt.legend()
plt.ylabel('Effect Size for Pathology (SDs)')
plt.show()
#%%
sns.set(rc={'figure.figsize':(7,4)})
cmap = plt.cm.get_cmap('Reds').copy()
cmap.set_bad('midnightblue')      # color of mask on heatmap
cmap.set_under('midnightblue') 
cycle_band_mat_beta = cycle_band_mat_coef.copy()
#cycle_band_mat_beta[cycle_band_mat_p_LR>0.05] = 0
mask= cycle_band_mat_p_LR>0.05
ax = sns.heatmap(cycle_band_mat_beta,yticklabels= fb_interest, xticklabels=fluctNames, annot=True,mask=mask, linewidths=1)
ax.patch.set(hatch='xx', edgecolor='black')
plt.grid(visible=False)
plt.title('')
plt.show()
#%%
mask= cycle_band_mat_p>0.05
ax = sns.heatmap(cycle_band_mat_beta,yticklabels= fb_interest, xticklabels=fluctNames, annot=True,mask=mask)
ax.patch.set(hatch='xx', edgecolor='black')
plt.grid(visible=False)
plt.title('is_soz Coefficient')
plt.show()

#plt.savefig(os.path.join ( plot_dir_all, 'mixed_reg_heatmap','soz_predict_cycle_power.pdf'.format(str.strip(cyc_name),var) ), bbox_inches='tight')
