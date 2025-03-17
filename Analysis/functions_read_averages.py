#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:33:56 2024

@author: 80021045
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import matplotlib
import math
from average_data_dep import *

from matplotlib.ticker import ScalarFormatter, NullFormatter
from scipy import stats
plt.rcParams.update({'font.size': 20})


def read_both0(pdiv_main_string, mutation_string_array):
    from average_data_dep import get_trajectories
    start_no_null = 1
    end_no_null = 250

    name_str_first_null = pdiv_main_string + \
        "/" + mutation_string_array + "/averages_"
    name_str_last_null = ".csv"

    all_dats_null = get_trajectories(
        name_str_first_null, name_str_last_null, start_no_null, end_no_null)

    return [all_dats_null, end_no_null]


def read_both44(pdiv_main_string, mutation_string_array):
    from average_data_dep import get_trajectories
    start_no_44 = 1
    end_no_44 = 250

    name_str_first_44 = pdiv_main_string + "/" + \
        mutation_string_array + "/averages_"
    name_str_last_44 = ".csv"

    all_dats_44 = get_trajectories(
        name_str_first_44, name_str_last_44, start_no_44, end_no_44)

    return [all_dats_44, end_no_44]


def read_both33(pdiv_main_string, mutation_string_array):
    from average_data_dep import get_trajectories
    start_no_33 = 1
    end_no_33 = 250

    name_str_first_33 = pdiv_main_string + "/" + \
        mutation_string_array + "/averages_"
    name_str_last_33 = ".csv"

    all_dats_33 = get_trajectories(
        name_str_first_33, name_str_last_33, start_no_33, end_no_33)
    data_33_no_cells = all_dats_33[0]

    return [all_dats_33, end_no_33]


def read_both22(pdiv_main_string, mutation_string_array):
    from average_data_dep import get_trajectories
    start_no_22 = 1
    end_no_22 = 150

    name_str_first_22 = pdiv_main_string + "/" + \
        mutation_string_array + "/averages_"
    name_str_last_22 = ".csv"

    all_dats_22 = get_trajectories(
        name_str_first_22, name_str_last_22, start_no_22, end_no_22)

    return [all_dats_22, end_no_22]


def read_both11(pdiv_main_string, mutation_string_array):
    from average_data_dep import get_trajectories
    start_no_11 = 1
    end_no_11 = 75

    name_str_first_11 = pdiv_main_string + "/" + \
        mutation_string_array + "/averages_"
    name_str_last_11 = ".csv"

    all_dats_11 = get_trajectories(
        name_str_first_11, name_str_last_11, start_no_11, end_no_11)

    return [all_dats_11, end_no_11]


def read_both00(pdiv_main_string, mutation_string_array):
    from average_data_dep import get_trajectories
    start_no_00 = 1
    end_no_00 = 35

    name_str_first_00 = pdiv_main_string + "/" + \
        mutation_string_array + "/averages_"
    name_str_last_00 = ".csv"

    all_dats_00 = get_trajectories(
        name_str_first_00, name_str_last_00, start_no_00, end_no_00)

    return [all_dats_00, end_no_00]

def FWD_Diff_Operator(data):
    short = len(data)
    diff = np.asarray(data[1::]) - np.asarray(data[0:(short-1)])
   
    return diff
#%% 
def FWD_Diff_Operator_Indv_Trajectories(data):
    all_diffs = []
    
    for i in range(len(data)):
        
        last_idx = len(data.loc[i])-1

        if (last_idx == 400):
        
            if (not (np.isnan(data.loc[i][400]))):
                all_diffs.append(FWD_Diff_Operator(data.loc[i]))
        else:
            all_diffs.append(np.array([0]*400))
                    
    return all_diffs
#%%     
def find_max_derivative(data_diffs):    
    if (len(data_diffs) >= 400):
        return data_diffs[np.argmax(np.abs(data_diffs))]
    else:
        return np.nan
#%%
def find_max_derivative_Indv_Trajectories(data_diffs):
    
    if (len(data_diffs) == 1):
        traj_diffs = data_diffs[0]
        all_max_diffs = traj_diffs[np.argmax(np.abs(traj_diffs))]
    else:
        all_max_diffs = []
    
        for i in range(len(data_diffs)):
            traj_diffs = data_diffs[i]
            all_max_diffs.append(traj_diffs[np.argmax(np.abs(traj_diffs))])

    return pd.Series(all_max_diffs)


#%% 

def find_time_and_mutations(indiv_fitness, end_no_sims, indiv_pos_muts):
    max_w = []
    t_max = []
    pos_muts_t_max = []
    for z in range(end_no_sims):
        w_t = indiv_fitness.loc[z]
        
       # if (len(w_t) >= 400):
        if (not (np.isnan(w_t[400]))):
            
            dw_t = FWD_Diff_Operator(w_t)
            maxw = dw_t[np.argmax(np.abs(dw_t))]
            ts_arg = np.argmax(np.abs(dw_t))
            pos_muts = indiv_pos_muts.loc[z]
            #the +1 is because it's a forward difference 
            max_w.append(maxw)
            t_max.append(ts_arg+1)
            pos_muts_t_max.append(pos_muts[ts_arg+1])
            
        
    
    if (len(t_max) > 0):
        
        combined_df = pd.DataFrame([pd.Series(max_w), pd.Series(t_max), pd.Series(pos_muts_t_max)]).T
        combined_df.columns = ['max_w','t_max', 'no_pos_muts'] 
    else:
        max_w.append(np.nan)
        t_max.append(np.nan)
        pos_muts_t_max(np.nan)
        
        combined_df = pd.DataFrame([pd.Series(max_w), pd.Series(t_max), pd.Series(pos_muts_t_max)]).T
        combined_df.columns = ['max_w','t_max', 'no_pos_muts'] 
    
    return combined_df
           


#%%
 #Variance of the no replicates 
def var_replicates(data, end_no, last_indx):
    var_arr = []
    incr_pt = []

    for z in range(end_no):
        if (len(data.loc[z]) >= (last_indx - 1)):
            pt = data.loc[z][last_indx - 1]
            if (not np.isnan(pt)):
                incr_pt.append(pt)
                var_arr.append(np.var(np.asarray(incr_pt)))
        else:
            z = z + 1
            
    return var_arr 

#%% 
def main_executive_function(pdiv_main_string, mutation_string_array, pdiv, output_directory, custom_cell_interval, custom_fitness_interval):

    
    plt.close('all') 
    
    w0_num = pdiv - 0.25
    string_pdiv = "p$_{div(0)}$ = %.2f" % pdiv
    dev0 = pdiv - 0.2/0.8
    string_dev0 = "W$_0$ = %.2f" % dev0
    string_pdiv_dev0 = string_pdiv + ", " + string_dev0

    time = np.linspace(1, 40001, 401)
    last_indx = 401

    start_no_null = 1
    start_no_44 = 1
    start_no_33 = 1
    start_no_22 = 1
    start_no_11 = 1
    start_no_00 = 1

    all_dats_null, end_no_null = read_both0(
        pdiv_main_string, mutation_string_array[0])
    all_dats_44, end_no_44 = read_both44(
        pdiv_main_string, mutation_string_array[1])
    all_dats_33, end_no_33 = read_both33(
        pdiv_main_string, mutation_string_array[2])
    all_dats_22, end_no_22 = read_both22(
        pdiv_main_string, mutation_string_array[3])
    all_dats_11, end_no_11 = read_both11(
        pdiv_main_string, mutation_string_array[4])
    all_dats_00, end_no_00 = read_both00(
        pdiv_main_string, mutation_string_array[5])

    #pmut = null
    data_null_no_cells = all_dats_null[0]
    data_null_pdiv = all_dats_null[1] / data_null_no_cells
    data_null_pdie = all_dats_null[2] / data_null_no_cells
    data_null_fit = data_null_pdiv - data_null_pdie

    data_null_pos_muts = all_dats_null[3] / data_null_no_cells
    data_null_neu_muts = all_dats_null[4] / data_null_no_cells
    data_null_neg_muts = all_dats_null[5] / data_null_no_cells
    data_null_diff_pos_muts = data_null_pos_muts - data_null_neg_muts

    #pmut = 1e-4
    data_44_no_cells = all_dats_44[0]
    data_44_pdiv = all_dats_44[1] / data_44_no_cells
    data_44_pdie = all_dats_44[2] / data_44_no_cells
    data_44_fit = data_44_pdiv - data_44_pdie

    data_44_pos_muts = all_dats_44[3] / data_44_no_cells
    data_44_neu_muts = all_dats_44[4] / data_44_no_cells
    data_44_neg_muts = all_dats_44[5] / data_44_no_cells
    data_44_diff_pos_muts = data_44_pos_muts - data_44_neg_muts

    #pmut = 1e-3
    data_33_no_cells = all_dats_33[0]

    data_33_pdiv = all_dats_33[1] / data_33_no_cells
    data_33_pdie = all_dats_33[2] / data_33_no_cells
    data_33_fit = data_33_pdiv - data_33_pdie

    data_33_pos_muts = all_dats_33[3] / data_33_no_cells
    data_33_neu_muts = all_dats_33[4] / data_33_no_cells
    data_33_neg_muts = all_dats_33[5] / data_33_no_cells
    data_33_diff_pos_muts = data_33_pos_muts - data_33_neg_muts

    #pmut = 1e-2
    data_22_no_cells = all_dats_22[0]
    data_22_pdiv = all_dats_22[1] / data_22_no_cells
    data_22_pdie = all_dats_22[2] / data_22_no_cells
    data_22_fit = data_22_pdiv - data_22_pdie

    data_22_pos_muts = all_dats_22[3] / data_22_no_cells
    data_22_neu_muts = all_dats_22[4] / data_22_no_cells
    data_22_neg_muts = all_dats_22[5] / data_22_no_cells
    data_22_diff_pos_muts = data_22_pos_muts - data_22_neg_muts

    #pmut = 1e-1
    data_11_no_cells = all_dats_11[0]
    data_11_pdiv = all_dats_11[1] / data_11_no_cells
    data_11_pdie = all_dats_11[2] / data_11_no_cells
    data_11_fit = data_11_pdiv - data_11_pdie

    data_11_pos_muts = all_dats_11[3] / data_11_no_cells
    data_11_neu_muts = all_dats_11[4] / data_11_no_cells
    data_11_neg_muts = all_dats_11[5] / data_11_no_cells
    data_11_diff_pos_muts = data_11_pos_muts - data_11_neg_muts

    #pmut = 1
    data_00_no_cells = all_dats_00[0]

    data_00_pdiv = all_dats_00[1] / data_00_no_cells
    data_00_pdie = all_dats_00[2] / data_00_no_cells
    data_00_fit = data_00_pdiv - data_00_pdie

    data_00_pos_muts = all_dats_00[3] / data_00_no_cells
    data_00_neu_muts = all_dats_00[4] / data_00_no_cells
    data_00_neg_muts = all_dats_00[5] / data_00_no_cells
    data_00_diff_pos_muts = data_00_pos_muts - data_00_neg_muts

    ### analyze how many cells survive ###

   
    #%% 
    ### analyze how many cells survive ###

    survival_null = analyze_survival(
        data_null_no_cells, end_no_null, last_indx)
    survival_44 = analyze_survival(data_44_no_cells, end_no_44, last_indx)
    survival_33 = analyze_survival(data_33_no_cells, end_no_33, last_indx)
    survival_22 = analyze_survival(data_22_no_cells, end_no_22, last_indx)
    survival_11 = analyze_survival(data_11_no_cells, end_no_11, last_indx)
    survival_00 = analyze_survival(data_00_no_cells, end_no_00, last_indx)

    # str_null = '$p_{mut}$ = 0, $\hat{n}$ = %d' % survival_null[1]
    # str_44 = '$p_{mut}$ = 1e-4, $\hat{n}$ = %d' % survival_44[1]
    # str_33 = '$p_{mut}$ = 1e-3, $\hat{n}$ = %d' % survival_33[1]
    # str_22 = '$p_{mut}$ = 1e-2, $\hat{n}$ = %d' % survival_22[1]
    # str_11 = '$p_{mut}$ = 1e-1, $\hat{n}$ = %d' % survival_11[1]
    # str_00 = '$p_{mut}$ = 1, $\hat{n}$ = %d' % survival_00[1]
    
    str_null = '$p_{mut}$ = 0'
    str_44 = '$p_{mut}$ = 1e-4'
    str_33 = '$p_{mut}$ = 1e-3'
    str_22 = '$p_{mut}$ = 1e-2'
    str_11 = '$p_{mut}$ = 1e-1'
    str_00 = '$p_{mut}$ = 1'

    ### SUPPLEMENTARY FIGURE: HOW MANY EXTINCT?

    survival_all = [survival_null[1], survival_44[1],
                    survival_33[1], survival_22[1], survival_11[1], survival_00[1]]
    per_survival_all = np.asarray(survival_all) / np.array(
        [end_no_null, end_no_44, end_no_33, end_no_22, end_no_11, end_no_00])

    colors = list(reversed(['darkred', 'darkorange', 'darkkhaki',
                  'forestgreen', 'royalblue', 'darkslateblue']))

    fig, ax = plt.subplots(1,  figsize=(11.5, 5))

    # creating the bar plot
    ax.bar(list(range(6)), per_survival_all * 100, color=colors,
           width=0.6)
    n_out_of_null_str = str(survival_null[1]) +  " of " + str(end_no_null)
    n_out_of_44_str = str(survival_44[1]) + " of " + str(end_no_44)
    n_out_of_33_str = str(survival_33[1]) +  " of " + str(end_no_33)
    n_out_of_22_str = str(survival_22[1]) +   " of " +str(end_no_22)
    n_out_of_11_str = str(survival_11[1]) +   " of " +str(end_no_11)
    n_out_of_00_str = str(survival_00[1]) +   " of " +str(end_no_00)
            
    ax.text(0, per_survival_all[0]*100 + 5, n_out_of_null_str, fontsize=20, ha='center')
    ax.text(1, per_survival_all[1]*100 + 5, n_out_of_44_str, fontsize=20, ha='center')
    ax.text(2, per_survival_all[2]*100 + 5, n_out_of_33_str, fontsize=20, ha='center')
    ax.text(3, per_survival_all[3]*100 + 5, n_out_of_22_str, fontsize=20, ha='center')
    ax.text(4, per_survival_all[4]*100 + 5, n_out_of_11_str, fontsize=20, ha='center')
    ax.text(5, per_survival_all[5]*100 + 5, n_out_of_00_str, fontsize=20, ha='center')

    ax.set_ylim(0, 115)
    ax.set_xticks([0, 1, 2, 3, 4, 5], list(
        reversed(['1', '1e-1', '1e-2', '1e-3', '1e-4', '0'])))  # , '0'
    ax.set_xlabel('probability of mutation $p_{mut}$', labelpad=20)
    ax.set_ylabel('Percentage (%)')
    main_title_string = 'Simulation survival frequency' #string_pdiv_dev0 + "\n " +  
    ax.set_title(main_title_string, y=1.05)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    
    
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(2)
    save_it = os.path.join(output_directory, "extinction_probability.pdf")

    plt.savefig(save_it, bbox_inches='tight')
    
    #################################
    
    ###
    ### POPULATION NUMBERS OVER TIME
    ### 
    
    #################################


    no_cells_null = calculate_error(data_null_no_cells)
    no_cells_44 = calculate_error(data_44_no_cells)
    no_cells_33 = calculate_error(data_33_no_cells)
    no_cells_22 = calculate_error(data_22_no_cells)
    no_cells_11 = calculate_error(data_11_no_cells)
    no_cells_00 = calculate_error(data_00_no_cells)

    # short corresponds to to time index of when to end the observation. 
    #Max is 401 corresponding to time[401] = 40000 / (365 * 2) = 54.79 years
    #Here we use 151 which is 20.68 years 

    for k in range(2):
        fig, ax = plt.subplots(1,figsize=(5, 5))
    
        short = np.min([last_indx, len(no_cells_null[0])])
        ax.plot(time[0:short],no_cells_null[0][0:short], '--.', linewidth = 3,color='darkslateblue', label = str_null )
        ax.fill_between(time[0:short], no_cells_null[1][0:short], no_cells_null[2][0:short][0:short], color = 'darkslateblue', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_44[0])])
        ax.plot(time[0:short],no_cells_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
        ax.fill_between(time[0:short], no_cells_44[1][0:short], no_cells_44[2][0:short], color = 'royalblue', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_33[0])])
        ax.plot(time[0:short],no_cells_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
        ax.fill_between(time[0:short], no_cells_33[1][0:short], no_cells_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_22[0])])
        ax.plot(time[0:short],no_cells_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
        ax.fill_between(time[0:short], no_cells_22[1][0:short], no_cells_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_11[0])])
        ax.plot(time[0:short], no_cells_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
        ax.fill_between(time[0:short], no_cells_11[1][0:short], no_cells_11[2][0:short], color = 'darkorange', alpha=0.4)
    
        short = np.min([last_indx, len(no_cells_00[0])])
        ax.plot(time[0:short], no_cells_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
        ax.fill_between(time[0:short], no_cells_00[1][0:short], no_cells_00[2][0:short], color = 'darkred', alpha=0.3)
    
    
        main_title_string =  'Average no. cells' #string_pdiv_dev0 + "\n " +
        ax.set_title(main_title_string, y=1.05)
    
        ax.set_ylabel('No. cells (log$_{10}$)')
        ax.set_xlabel('time [steps] (log$_{10}$)',  labelpad=20)
    
        ax.set_xscale('log')
        ax.set_yscale('log')
    
        ax.set_xticks([1, 731, 7301, 36501], [0, 1, 10, 50], fontsize = 20)
        ax.set_yticks([50,1000,10000], [50,1000,10000])
        # for axis in ['top','bottom','left','right']:
        #     ax.spines[axis].set_linewidth(2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=2)
    
        save_it = os.path.join(output_directory, "no_cells_no_legend.pdf")
        #legend
        if (k == 1):
            plt.legend(bbox_to_anchor=(1.07,0.9), borderaxespad=0,frameon=False, fontsize = 18) 
    
            save_it = os.path.join(output_directory, "no_cells_with_legend.pdf")
    
        plt.savefig(save_it, bbox_inches='tight')
        
        
        #%%
        
       


        var_null = var_replicates(data_null_no_cells, end_no_null, last_indx)
        var_44 = var_replicates(data_44_no_cells, end_no_44, last_indx)
        var_33 = var_replicates(data_33_no_cells, end_no_33, last_indx)
        var_22 = var_replicates(data_22_no_cells, end_no_22, last_indx)
        var_11 = var_replicates(data_11_no_cells, end_no_11, last_indx)
        var_00 = var_replicates(data_00_no_cells, end_no_00, last_indx)    

        
        
        #fig, ax = plt.subplots(1,2,figsize=(13,5.5))
        fig, ax = plt.subplots(1,figsize=(5,5.5))
        fig.subplots_adjust(wspace=0.4)

        ax.plot(var_null, '--.', linewidth = 3,color='darkslateblue', label = str_null)
        ax.plot(var_44, '--.', linewidth = 3,color='royalblue', label = str_44)
        ax.plot(var_33, '-.', linewidth = 3,color='forestgreen', label = str_33)
        ax.plot(var_22, ':', linewidth = 3,color='darkkhaki', label = str_22)
        ax.plot(var_11,  '-', linewidth = 3,color='darkorange', label = str_11)
        ax.plot(var_00, '--', linewidth = 3,color='darkred', label = str_00)

        ax.set_xlabel('no. simulations')
        ax.set_ylabel('Variance in no. cells')



        plt.legend(bbox_to_anchor=(1.07,0.9), borderaxespad=0,frameon=False, fontsize = 18) 
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.suptitle('Variance over the no. simulations', y = 1.005)
        
        save_it = os.path.join(output_directory, "variance_no_sim.pdf")

        plt.savefig(save_it, bbox_inches='tight')

        
    
    
    #################################
    
    ###
    ### Cell fitness over time 
    ### 
    
    ### TOEDIT
    # main_title_string = string_pdiv_dev0 + "\n " + \
    #     'Average no. cells'
    # ax.set_title(main_title_string, y=1.05)
    # save_it = os.path.join(output_directory, "no_cells.pdf")
    # plt.savefig(save_it, bbox_inches='tight')
    
    #################################
    
    
    fit_null = calculate_error(data_null_fit)
    fit_44 = calculate_error(data_44_fit)
    fit_33 = calculate_error(data_33_fit)
    fit_22 = calculate_error(data_22_fit)
    fit_11 = calculate_error(data_11_fit)
    fit_00 = calculate_error(data_00_fit)

    fit_low, fit_high = custom_fitness_interval
    
    
    


    for k in range(2):
        fig, ax = plt.subplots(1,figsize=(5.5,5.5))
    
        short = np.min([last_indx, len(no_cells_null[0])])
        ax.plot(time[0:short],fit_null[0][0:short], '--.', linewidth = 3,color='darkslateblue', label = str_null )
        ax.fill_between(time[0:short], fit_null[1][0:short], fit_null[2][0:short][0:short], color = 'darkslateblue', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_44[0])])
        ax.plot(time[0:short],fit_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
        ax.fill_between(time[0:short], fit_44[1][0:short], fit_44[2][0:short], color = 'royalblue', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_33[0])])
        ax.plot(time[0:short],fit_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
        ax.fill_between(time[0:short], fit_33[1][0:short], fit_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_22[0])])
        ax.plot(time[0:short],fit_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
        ax.fill_between(time[0:short], fit_22[1][0:short], fit_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_11[0])])
        ax.plot(time[0:short], fit_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
        ax.fill_between(time[0:short], fit_11[1][0:short], fit_11[2][0:short], color = 'darkorange', alpha=0.4)
    
        short = np.min([last_indx, len(no_cells_00[0])])
        ax.plot(time[0:short], fit_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
        ax.fill_between(time[0:short], fit_00[1][0:short], fit_00[2][0:short], color = 'darkred', alpha=0.3)
    
        
        ax.set_xlabel('time [steps] (log$_{10}$)')
        ax.set_ylabel(r"$\bar{w}$ (p$_{div}$ - p$_{die}$)", x = -0.05)
    
        
        ax.set_xscale('log')
        ax.set_xticks([1, 731, 7301, 36501], [0, 1, 10, 50], fontsize = 20)
        
        # for axis in ['top','bottom','left','right']:
        #     ax.spines[axis].set_linewidth(2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(width=2)
        ax.set_ylim(0,1)
        main_title_string = 'Average fitness' #string_pdiv_dev0 + "\n " + 
        ax.set_title(main_title_string, y=1.05)
        save_it = os.path.join(output_directory, "fitness.pdf")
        
        if (k == 1): 
            ax.set_ylim(fit_low, fit_high)
            
            save_it = os.path.join(output_directory, "fitness_zoom.pdf")
        plt.savefig(save_it, bbox_inches='tight')
    plt.close('all')        
    
    ######### 
    # fitness by the end of time 
    #########    
    last_indx_0 = np.min([last_indx, len(no_cells_null[0])])    
    last_indx_null = np.min([last_indx, len(no_cells_null[0])])
    last_indx_44 = np.min([last_indx, len(no_cells_44[0])])
    last_indx_33 = np.min([last_indx, len(no_cells_33[0])])
    last_indx_22 = np.min([last_indx, len(no_cells_22[0])])
    last_indx_11 = np.min([last_indx, len(no_cells_11[0])])
    last_indx_00 = np.min([last_indx, len(no_cells_00[0])])
    
        

    data_fit_last_null = np.asarray(data_null_fit[last_indx_null-1][~np.isnan(data_null_fit[last_indx_null-1])]) 
    data_fit_last_44 = np.asarray(data_44_fit[last_indx_44-1][~np.isnan(data_44_fit[last_indx_44-1])])
    data_fit_last_33 = np.asarray(data_33_fit[last_indx_33-1][~np.isnan(data_33_fit[last_indx_33-1])])                                   
    data_fit_last_22 = np.asarray(data_22_fit[last_indx_22-1][~np.isnan(data_22_fit[last_indx_22-1])])
    data_fit_last_11 = np.asarray(data_11_fit[last_indx_11-1][~np.isnan(data_11_fit[last_indx_11-1])])   
    data_fit_last_00 = np.asarray(data_00_fit[last_indx_00-1][~np.isnan(data_00_fit[last_indx_00-1])])       


    data_no_cells_last_null = np.asarray(data_null_no_cells[last_indx_null-1][~np.isnan(data_null_no_cells[last_indx_null-1])]) 
    data_no_cells_last_44 = np.asarray(data_44_no_cells[last_indx_44-1][~np.isnan(data_44_no_cells[last_indx_44-1])])
    data_no_cells_last_33 = np.asarray(data_33_no_cells[last_indx_33-1][~np.isnan(data_33_no_cells[last_indx_33-1])])                                   
    data_no_cells_last_22 = np.asarray(data_22_no_cells[last_indx_22-1][~np.isnan(data_22_no_cells[last_indx_22-1])])
    data_no_cells_last_11 = np.asarray(data_11_no_cells[last_indx_11-1][~np.isnan(data_11_no_cells[last_indx_11-1])])   
    data_no_cells_last_00 = np.asarray(data_00_no_cells[last_indx_00-1][~np.isnan(data_00_no_cells[last_indx_00-1])]) 


    stats_null = pd.Series([last_indx_null, np.mean(data_no_cells_last_null), np.var(data_no_cells_last_null), np.mean(data_fit_last_null), np.var(data_fit_last_null)])
    stats_44 = pd.Series([last_indx_44, np.mean(data_no_cells_last_44), np.var(data_no_cells_last_44), np.mean(data_fit_last_44), np.var(data_fit_last_44)])
    stats_33 = pd.Series([last_indx_33, np.mean(data_no_cells_last_33), np.var(data_no_cells_last_33), np.mean(data_fit_last_33), np.var(data_fit_last_33)])
    stats_22 = pd.Series([last_indx_22, np.mean(data_no_cells_last_22), np.var(data_no_cells_last_22), np.mean(data_fit_last_22), np.var(data_fit_last_22)])
    stats_11 = pd.Series([last_indx_11, np.mean(data_no_cells_last_11), np.var(data_no_cells_last_11), np.mean(data_fit_last_11), np.var(data_fit_last_11)])
    stats_00 = pd.Series([last_indx_00, np.mean(data_no_cells_last_00), np.var(data_no_cells_last_00), np.mean(data_fit_last_00), np.var(data_fit_last_00)])


    stats_last = pd.DataFrame([stats_null,stats_44, stats_33, stats_22, stats_11, stats_00])
    stats_last.columns = ["last_indx", "mean_no_cells", "var_no_cells", "mean_fitness", "var_fitness"]
    
    save_it = os.path.join(output_directory, "stats_last.pkl")
    stats_last.to_pickle(save_it)  


    fit_last = [ data_null_fit[last_indx_null-1][~np.isnan(data_null_fit[last_indx_null-1])],
                data_44_fit[last_indx_44-1][~np.isnan(data_44_fit[last_indx_44-1])],
                data_33_fit[last_indx_33-1][~np.isnan(data_33_fit[last_indx_33-1])],
                data_22_fit[last_indx_22-1][~np.isnan(data_22_fit[last_indx_22-1])],
                data_11_fit[last_indx_11-1][~np.isnan(data_11_fit[last_indx_11-1])],
                data_00_fit[last_indx_00-1][~np.isnan(data_00_fit[last_indx_00-1])]]

    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6.5,5), sharex = True, gridspec_kw={'height_ratios': [8, 2]}) # gridspec_kw={'width_ratios': [5, 4]}
    fig.subplots_adjust(hspace=0.25, left = 0.2, top =0.85, bottom = 0.01)

    boxplt2 = ax1.boxplot(fit_last , showfliers=False , patch_artist=True) 
    boxplt3 = ax2.boxplot(fit_last , showfliers=False , patch_artist=True) 


    for median in boxplt2['medians']:
        median.set_color('black')
        
    for median in boxplt3['medians']:
        median.set_color('black') 

    colors = list(reversed(['darkred', 'darkorange', 'darkkhaki', 'forestgreen', 'royalblue', 'darkslateblue']))


    for patch, color in zip(boxplt2['boxes'], colors):
        patch.set_facecolor(color)
        #patch.set_alpha(0.75)

    for patch, color in zip(boxplt3['boxes'], colors):
        patch.set_facecolor(color)
        #patch.set_alpha(0.75)


   
    # # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(fit_low, fit_high)  # outliers only
    ax2.set_ylim(0, 0.2)  # most of the data


    ax1.set_yticks([fit_low, fit_high]) # , fontsize = 18
    ax2.set_yticks([0, 0.2]) #, fontsize = 18


    ax2.set_xticks([1,2,3,4,5, 6], list(reversed(['1', '1e-1', '1e-2', '1e-3', '1e-4', '0'])))
    ax2.set_xlabel('probability of mutation $p_{mut}$', labelpad = 10) #, fontsize = 18

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax2.spines.right.set_visible(False)
    ax1.spines.right.set_visible(False)

    #ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.tick_params(labeltop=False)

    #ax2.xaxis.tick_bottom()
    ax2.set_xlabel('$p_{mut}$', fontsize = 22) #, fontsize = 18
    fig.supylabel("$w$", fontsize = 22) #, fontsize = 18


    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    
    
    main_title_string = 'Average fitness at t = t$_{end}$ for individual simulations' #string_pdiv_dev0 + "\n " + 
   # fig.suptitle(main_title_string, y=1.05)
    save_it = os.path.join(output_directory, "fitness_boxplots1.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    
    #PART 2: zoom in 

    fig, ax1 = plt.subplots(1,figsize=(5,5), sharex = True) # gridspec_kw={'width_ratios': [5, 4]}
    fig.subplots_adjust(hspace=0.25, left = 0.2, top =0.85, bottom = 0.01)

    boxplt2 = ax1.boxplot([fit_last[1], fit_last[2], fit_last[3], fit_last[4]] , showfliers=False , patch_artist=True ) 
    color1 = ['royalblue', 'forestgreen',  'darkkhaki', 'darkorange' ]

    for median in boxplt2['medians']:
        median.set_color('black')
        

    for patch, color in zip(boxplt2['boxes'], color1):
        patch.set_facecolor(color)

    ax1.set_xticks([1,2,3,4], list(['1e-4', '1e-3', '1e-2', '1e-1']))
    ax1.set_xlabel('$p_{mut}$', labelpad = 10) #, fontsize = 18
    ax1.spines.top.set_visible(False)
    ax1.spines.right.set_visible(False)
    ax1.set_ylabel("$w$", fontsize = 22) #, fontsize = 18
    #ax1.set_title('Average fitness at t = t$_{end}$\n', fontsize = 24)

    main_title_string = 'Average fitness at t = t$_{end}$'
    #ax1.set_title(main_title_string, y=1.05)
    save_it = os.path.join(output_directory, "fitness_boxplots2.pdf")
    plt.savefig(save_it, bbox_inches='tight')



#%% 
#Max fitness differences  
          
        
    fwd_diff_fit_traj_00 = FWD_Diff_Operator_Indv_Trajectories(data_00_fit)   
    max_diff_00 = find_max_derivative_Indv_Trajectories(fwd_diff_fit_traj_00)   
    
    fwd_diff_fit_traj_11 = FWD_Diff_Operator_Indv_Trajectories(data_11_fit)   
    max_diff_11 = find_max_derivative_Indv_Trajectories(fwd_diff_fit_traj_11) 

    fwd_diff_fit_traj_22 = FWD_Diff_Operator_Indv_Trajectories(data_22_fit)   
    max_diff_22 = find_max_derivative_Indv_Trajectories(fwd_diff_fit_traj_22) 

    fwd_diff_fit_traj_33 = FWD_Diff_Operator_Indv_Trajectories(data_33_fit)   
    max_diff_33 = find_max_derivative_Indv_Trajectories(fwd_diff_fit_traj_33)   

    fwd_diff_fit_traj_44 = FWD_Diff_Operator_Indv_Trajectories(data_44_fit)   
    max_diff_44 = find_max_derivative_Indv_Trajectories(fwd_diff_fit_traj_44) 

#%%
    try_max_diffs = [max_diff_44, max_diff_33, max_diff_22, max_diff_11, max_diff_00]
    

    fig, ax = plt.subplots(1, figsize = (5,5))

    boxplt2 = ax.boxplot(try_max_diffs, showfliers=False , patch_artist=True) 
     
    for median in boxplt2['medians']:
        median.set_color('black')
        

    colors = list(reversed(['darkred', 'darkorange', 'darkkhaki', 'forestgreen', 'royalblue'])) #, 'darkslateblue'


    for patch, color in zip(boxplt2['boxes'], colors):
        patch.set_facecolor(color)
        #patch.set_alpha(0.75)
    ax.set_xticks([1,2,3,4,5], list(reversed(['1', '1e-1', '1e-2', '1e-3', '1e-4'])))
    ax.set_xlabel('$p_{mut}$', labelpad = 10)
    ax.set_ylabel('Max $\Delta$w', labelpad = 10)
      #, fontsize = 18
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Max $\Delta$w for \nindividual simulations', y = 1.05)
    save_it = os.path.join(output_directory, "max_Deltaw_indiv.pdf")
    plt.savefig(save_it, bbox_inches='tight')


    m = 0
    fig, ax = plt.subplots(2,3, figsize = (15,10), sharex = True, sharey=True)
    

    #for m in range(len(fwd_diff_fit_traj_44)):
    ax[0,1].plot(fwd_diff_fit_traj_44[m], linewidth = 3, color = 'royalblue')
    
    #for m in range(len(fwd_diff_fit_traj_33)):
    ax[0,2].plot(fwd_diff_fit_traj_33[m], linewidth = 3, color = 'forestgreen')    
    
    #for m in range(len(fwd_diff_fit_traj_22)):
    ax[1,0].plot(fwd_diff_fit_traj_22[m], linewidth = 3, color = 'darkkhaki')    
    
    #for m in range(len(fwd_diff_fit_traj_11)):
    ax[1,1].plot(fwd_diff_fit_traj_11[m], linewidth = 3, color = 'darkorange')      
    
    #for m in range(len(fwd_diff_fit_traj_11)):
    ax[1,2].plot(fwd_diff_fit_traj_00[m], linewidth = 3, color = 'darkred')     
    
    ax[0,0].set_xscale('log')
    ax[0,1].set_xscale('log')
    ax[0,2].set_xscale('log')
    ax[1,0].set_xscale('log')
    ax[1,1].set_xscale('log')
    ax[1,2].set_xscale('log')
   
   
   
    ax[0,0].set_title('p$_{mut}$ = 0 (nothing to plot)')
    ax[0,1].set_title('p$_{mut}$ = 1e-4')
    ax[0,2].set_title('p$_{mut}$ = 1e-3')
    ax[1,0].set_title('p$_{mut}$ = 1e-2')
    ax[1,1].set_title('p$_{mut}$ = 1e-1')
    ax[1,2].set_title('p$_{mut}$ = 1')
    
    fig.suptitle('Examples of $\Delta$w for individual simulations', y = 1.005, fontsize = 24)
    fig.supxlabel('time step (log10)', y = -0.02)
    fig.supylabel('$\Delta$w', x = 0.07)
    
    save_it = os.path.join(output_directory, "individual_max_deltaw.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    




#%% 


#%% 

    data_dev_null = data_null_pdiv - data_null_pdie/(1-data_null_pdie)
    data_dev_44 = data_44_pdiv - data_44_pdie/(1-data_44_pdie)
    data_dev_33 = data_33_pdiv - data_33_pdie/(1-data_33_pdie)
    data_dev_22 = data_22_pdiv - data_22_pdie/(1-data_22_pdie)
    data_dev_11 = data_11_pdiv - data_11_pdie/(1-data_11_pdie)
    data_dev_00 = data_00_pdiv - data_00_pdie/(1-data_00_pdie)

    dev_null = calculate_error(data_dev_null)
    dev_44 = calculate_error(data_dev_44)
    dev_33 = calculate_error(data_dev_33)
    dev_22 = calculate_error(data_dev_22)
    dev_11 = calculate_error(data_dev_11)
    dev_00 = calculate_error(data_dev_00)
    
  
    dev_high = 0.757
    dev_low = 0.727

    for k in range(2):
        fig, ax = plt.subplots(1,figsize=(5.5,5.5))
    
        short = np.min([last_indx, len(no_cells_null[0])])
        ax.plot(time[0:short],dev_null[0][0:short], '--.', linewidth = 3,color='darkslateblue', label = str_null )
        ax.fill_between(time[0:short], dev_null[1][0:short], dev_null[2][0:short][0:short], color = 'darkslateblue', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_44[0])])
        ax.plot(time[0:short],dev_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
        ax.fill_between(time[0:short], dev_44[1][0:short], dev_44[2][0:short], color = 'royalblue', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_33[0])])
        ax.plot(time[0:short],dev_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
        ax.fill_between(time[0:short], dev_33[1][0:short], dev_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_22[0])])
        ax.plot(time[0:short],dev_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
        ax.fill_between(time[0:short], dev_22[1][0:short], dev_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
        short = np.min([last_indx, len(no_cells_11[0])])
        ax.plot(time[0:short], dev_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
        ax.fill_between(time[0:short], dev_11[1][0:short], dev_11[2][0:short], color = 'darkorange', alpha=0.4)
    
        short = np.min([last_indx, len(no_cells_00[0])])
        ax.plot(time[0:short], dev_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
        ax.fill_between(time[0:short], dev_00[1][0:short], dev_00[2][0:short], color = 'darkred', alpha=0.3)
    
        
        ax.set_xlabel('time [steps] (log$_{10}$)')
        ax.set_ylabel('p$_{div}$ - p$_{die}$/(1-p$_{die}$)')
    
        
        ax.set_xscale('log')
        ax.set_xticks([1, 731, 7301, 36501], [0, 1, 10, 50], fontsize = 20)
        
        # for axis in ['top','bottom','left','right']:
        #     ax.spines[axis].set_linewidth(2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        
        ax.tick_params(width=2)
        ax.set_ylim(0,1)
        main_title_string = 'Average Dev$_{0}$' #string_pdiv_dev0 + "\n " + 
        ax.set_title(main_title_string, y=1.05)
        save_it = os.path.join(output_directory, "dev.pdf")
        
        if (k == 1): 
            ax.set_ylim(dev_low, dev_high)
            
            save_it = os.path.join(output_directory, "dev_zoom.pdf")
        plt.savefig(save_it, bbox_inches='tight')


    plt.close('all')




# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)


#%% 

    ### 
    """ 
    POSITIVE AND NEGATIVE MUTATIONS OVER TIME 
    
    """ 
    ### 
    

    pos_muts_null = calculate_error(data_null_pos_muts)
    pos_muts_44 = calculate_error(data_44_pos_muts)
    pos_muts_33 = calculate_error(data_33_pos_muts)
    pos_muts_22 = calculate_error(data_22_pos_muts)
    pos_muts_11 = calculate_error(data_11_pos_muts)
    pos_muts_00 = calculate_error(data_00_pos_muts)
    
    neg_muts_null = calculate_error(data_null_neg_muts)
    neg_muts_44 = calculate_error(data_44_neg_muts)
    neg_muts_33 = calculate_error(data_33_neg_muts)
    neg_muts_22 = calculate_error(data_22_neg_muts)
    neg_muts_11 = calculate_error(data_11_neg_muts)
    neg_muts_00 = calculate_error(data_00_neg_muts)
    
    
    diff_pos_muts_44 = calculate_error(data_44_diff_pos_muts)
    diff_pos_muts_33 = calculate_error(data_33_diff_pos_muts)
    diff_pos_muts_22 = calculate_error(data_22_diff_pos_muts)
    diff_pos_muts_11 = calculate_error(data_11_diff_pos_muts)
    diff_pos_muts_00 = calculate_error(data_00_diff_pos_muts)
    
    fig, ax = plt.subplots(1,3,figsize=(21.6,5.4), sharey = True, sharex = True)
    fig.subplots_adjust(wspace = 0.5, hspace = 5) #bottom=0.1,
    ax[0].set_yscale('log')
    
    short = np.min([last_indx, len(no_cells_44[0])])
    ax[0].plot(time[0:short],pos_muts_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
    ax[0].fill_between(time[0:short], pos_muts_44[1][0:short], pos_muts_44[2][0:short], color = 'royalblue', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_33[0])])
    ax[0].plot(time[0:short],pos_muts_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
    ax[0].fill_between(time[0:short], pos_muts_33[1][0:short], pos_muts_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_22[0])])
    ax[0].plot(time[0:short],pos_muts_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
    ax[0].fill_between(time[0:short], pos_muts_22[1][0:short], pos_muts_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_11[0])])
    ax[0].plot(time[0:short], pos_muts_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
    ax[0].fill_between(time[0:short], pos_muts_11[1][0:short], pos_muts_11[2][0:short], color = 'darkorange', alpha=0.4)
    
    short = np.min([last_indx, len(no_cells_00[0])])
    ax[0].plot(time[0:short], pos_muts_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
    ax[0].fill_between(time[0:short], pos_muts_00[1][0:short], pos_muts_00[2][0:short], color = 'darkred', alpha=0.3)
    
    ax[0].axhline(y=1, color = 'black', linewidth = 2, linestyle = '--')
    
    
    ax[0].set_xscale('log')
    ax[0].set_xticks([1, 731, 7301, 36501], [0, 1, 10, 50], fontsize = 20)
    
    ax[0].set_ylim(0.005,500)

    ax[0].set_ylabel('no. (+) mutations')
    
    
    
    ax[0].set_title('Average no. positive \n mutations', pad=30)
    
    short = np.min([last_indx, len(no_cells_44[0])])
    ax[1].plot(time[0:short],neg_muts_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
    ax[1].fill_between(time, neg_muts_44[1], neg_muts_44[2], color = 'royalblue', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_33[0])])
    ax[1].plot(time[0:short],neg_muts_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
    ax[1].fill_between(time[0:short], neg_muts_33[1][0:short], neg_muts_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_22[0])])
    ax[1].plot(time[0:short],neg_muts_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
    ax[1].fill_between(time[0:short], neg_muts_22[1][0:short], neg_muts_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_11[0])])
    ax[1].plot(time[0:short], neg_muts_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
    ax[1].fill_between(time[0:short], neg_muts_11[1][0:short], neg_muts_11[2][0:short], color = 'darkorange', alpha=0.4)
    
    short = np.min([last_indx, len(no_cells_00[0])])
    ax[1].plot(time[0:short], neg_muts_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
    ax[1].fill_between(time[0:short], neg_muts_00[1][0:short], neg_muts_00[2][0:short], color = 'darkred', alpha=0.3)
    
    
    ax[1].axhline(y=1, color = 'black', linewidth = 2, linestyle = '--')
    
    

    ax[1].set_ylabel('no. (-) mutations')
    
    
    ax[1].set_title('Average no. negative mutations', pad=30)
    
    ax[1].set_yscale('log')
    
    
    
    short = np.min([last_indx, len(no_cells_44[0])])
    ax[2].plot(time[0:short],diff_pos_muts_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
    ax[2].fill_between(time[0:short], diff_pos_muts_44[1][0:short], diff_pos_muts_44[2][0:short], color = 'royalblue', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_33[0])])
    ax[2].plot(time[0:short],diff_pos_muts_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
    ax[2].fill_between(time[0:short], diff_pos_muts_33[1][0:short], diff_pos_muts_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_22[0])])
    ax[2].plot(time[0:short],diff_pos_muts_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
    ax[2].fill_between(time[0:short], diff_pos_muts_22[1][0:short], diff_pos_muts_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_11[0])])
    ax[2].plot(time[0:short], diff_pos_muts_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
    ax[2].fill_between(time[0:short], diff_pos_muts_11[1][0:short], diff_pos_muts_11[2][0:short], color = 'darkorange', alpha=0.4)
    
    short = np.min([last_indx, len(no_cells_00[0])])
    ax[2].plot(time[0:short], diff_pos_muts_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
    ax[2].fill_between(time[0:short], diff_pos_muts_00[1][0:short], diff_pos_muts_00[2][0:short], color = 'darkred', alpha=0.3)
    
    ax[2].set_yscale('log')
    
    

    ax[2].set_ylabel('(+) - (-) mutations')
    
    
    ax[2].set_title('Average differences between\n the no. positive and negative mutations', pad=30)
    
    ax[2].axhline(y=1, color = 'black', linewidth = 2, linestyle = '--')
    fig.supxlabel('time [steps] (log$_{10}$)', y = -0.05)

    
    for q in range(3):
        ax[q].spines['top'].set_visible(False)
        ax[q].spines['right'].set_visible(False)
    
    
    # main_title_string = string_pdiv_dev0 + "\n " + ''
    # fig.suptitle(main_title_string, y=1.25)
    
    save_it = os.path.join(output_directory, "average_no_mutations.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    #%% 
    fig, ax = plt.subplots(1,2,figsize=(9.72,5.4), sharey = True, sharex = True)
    fig.subplots_adjust(wspace = 0.5, hspace = 5) #bottom=0.1,
    ax[0].set_yscale('log')
    
    short = np.min([last_indx, len(no_cells_44[0])])
    ax[0].plot(time[0:short],pos_muts_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
    ax[0].fill_between(time[0:short], pos_muts_44[1][0:short], pos_muts_44[2][0:short], color = 'royalblue', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_33[0])])
    ax[0].plot(time[0:short],pos_muts_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
    ax[0].fill_between(time[0:short], pos_muts_33[1][0:short], pos_muts_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_22[0])])
    ax[0].plot(time[0:short],pos_muts_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
    ax[0].fill_between(time[0:short], pos_muts_22[1][0:short], pos_muts_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_11[0])])
    ax[0].plot(time[0:short], pos_muts_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
    ax[0].fill_between(time[0:short], pos_muts_11[1][0:short], pos_muts_11[2][0:short], color = 'darkorange', alpha=0.4)
    
    short = np.min([last_indx, len(no_cells_00[0])])
    ax[0].plot(time[0:short], pos_muts_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
    ax[0].fill_between(time[0:short], pos_muts_00[1][0:short], pos_muts_00[2][0:short], color = 'darkred', alpha=0.3)
    
    ax[0].axhline(y=1, color = 'black', linewidth = 2, linestyle = '--')
    
    
    ax[0].set_xscale('log')
    ax[0].set_xticks([1, 731, 7301, 36501], [0, 1, 10, 50], fontsize = 20)
    
    ax[0].set_ylim(0.005,500)
    ax[0].set_xlabel('time (years)')
    
    
    
    
    ax[0].set_title('positive', pad=30)
    
    short = np.min([last_indx, len(no_cells_44[0])])
    ax[1].plot(time[0:short],neg_muts_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
    ax[1].fill_between(time, neg_muts_44[1], neg_muts_44[2], color = 'royalblue', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_33[0])])
    ax[1].plot(time[0:short],neg_muts_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
    ax[1].fill_between(time[0:short], neg_muts_33[1][0:short], neg_muts_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_22[0])])
    ax[1].plot(time[0:short],neg_muts_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
    ax[1].fill_between(time[0:short], neg_muts_22[1][0:short], neg_muts_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_11[0])])
    ax[1].plot(time[0:short], neg_muts_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
    ax[1].fill_between(time[0:short], neg_muts_11[1][0:short], neg_muts_11[2][0:short], color = 'darkorange', alpha=0.4)
    
    short = np.min([last_indx, len(no_cells_00[0])])
    ax[1].plot(time[0:short], neg_muts_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
    ax[1].fill_between(time[0:short], neg_muts_00[1][0:short], neg_muts_00[2][0:short], color = 'darkred', alpha=0.3)
    
    
    ax[1].axhline(y=1, color = 'black', linewidth = 2, linestyle = '--')
    
    

    ax[1].set_title('negative', pad=20)
    
    ax[1].set_yscale('log')
    
    fig.supylabel('no. mutations', x = -0.04)
    fig.suptitle('Average no. mutations', y = 1.1)
   
    
    for q in range(2):
        ax[q].spines['top'].set_visible(False)
        ax[q].spines['right'].set_visible(False)
    
    
    main_title_string = string_pdiv_dev0 + "\n " + ''
    fig.suptitle(main_title_string, y=1.25)
    
    save_it = os.path.join(output_directory, "shorter_average_no_mutations.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    
    
    #%% Shorter, separate include only positive and negative mutations 
    
    fig, ax = plt.subplots(1,figsize=(7,5), sharey = True, sharex = True)
    fig.subplots_adjust(wspace = 0.5, hspace = 5) #bottom=0.1,
    ax.set_yscale('log')
    
    short = np.min([last_indx, len(no_cells_44[0])])
    ax.plot(time[0:short],pos_muts_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
    ax.fill_between(time[0:short], pos_muts_44[1][0:short], pos_muts_44[2][0:short], color = 'royalblue', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_33[0])])
    ax.plot(time[0:short],pos_muts_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
    ax.fill_between(time[0:short], pos_muts_33[1][0:short], pos_muts_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_22[0])])
    ax.plot(time[0:short],pos_muts_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
    ax.fill_between(time[0:short], pos_muts_22[1][0:short], pos_muts_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_11[0])])
    ax.plot(time[0:short], pos_muts_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
    ax.fill_between(time[0:short], pos_muts_11[1][0:short], pos_muts_11[2][0:short], color = 'darkorange', alpha=0.4)
    
    short = np.min([last_indx, len(no_cells_00[0])])
    ax.plot(time[0:short], pos_muts_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
    ax.fill_between(time[0:short], pos_muts_00[1][0:short], pos_muts_00[2][0:short], color = 'darkred', alpha=0.3)
    
    ax.axhline(y=1, color = 'black', linewidth = 2, linestyle = '--')
    
    
    ax.set_xscale('log')
    ax.set_xticks([1, 731, 7301, 36501], [0, 1, 10, 50], fontsize = 20)
    
    ax.set_ylim(0.005,500)
    ax.set_xlabel('time (years)')
    ax.set_ylabel('no. (+) mutations')
    
    
    
    ax.set_title('Average no. positive \n mutations', pad=30)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    main_title_string = string_pdiv_dev0 + "\n " + ''
    fig.suptitle(main_title_string, y=1.25)
    
    save_it = os.path.join(output_directory, "average_positive_mutations_nodiffs.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    
    
    
    
    fig, ax = plt.subplots(1,figsize=(7,5), sharey = True, sharex = True)
    fig.subplots_adjust(wspace = 0.5, hspace = 5) #bottom=0.1,
    ax.set_yscale('log')
    short = np.min([last_indx, len(no_cells_44[0])])
    ax.plot(time[0:short],neg_muts_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
    ax.fill_between(time, neg_muts_44[1], neg_muts_44[2], color = 'royalblue', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_33[0])])
    ax.plot(time[0:short],neg_muts_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
    ax.fill_between(time[0:short], neg_muts_33[1][0:short], neg_muts_33[2][0:short], color = 'forestgreen', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_22[0])])
    ax.plot(time[0:short],neg_muts_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
    ax.fill_between(time[0:short], neg_muts_22[1][0:short], neg_muts_22[2][0:short], color = 'darkkhaki', alpha=0.3)
    
    short = np.min([last_indx, len(no_cells_11[0])])
    ax.plot(time[0:short], neg_muts_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
    ax.fill_between(time[0:short], neg_muts_11[1][0:short], neg_muts_11[2][0:short], color = 'darkorange', alpha=0.4)
    
    short = np.min([last_indx, len(no_cells_00[0])])
    ax.plot(time[0:short], neg_muts_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
    ax.fill_between(time[0:short], neg_muts_00[1][0:short], neg_muts_00[2][0:short], color = 'darkred', alpha=0.3)
    
    
    ax.axhline(y=1, color = 'black', linewidth = 2, linestyle = '--')
    
    
    ax.set_xlabel('time (years)', y = -0.05)
    ax.set_ylabel('no. (-) mutations')
    
    
    ax.set_title('Average no. negative mutations', pad=30)
    

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    main_title_string = string_pdiv_dev0 + "\n " + ''
    fig.suptitle(main_title_string, y=1.25)
    
    save_it = os.path.join(output_directory, "average_negative_mutations_nodiffs.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    
    
    #%% 

    
   
    ### ZOOM AROUND CARRYING CAPACITY ### 
    
    fig, ax = plt.subplots(1,figsize=(5,5))
    fig.subplots_adjust(wspace=0.5)


    ### TRAJECTORIES OVER VERY LONG TIME SCALE 
    
    
   
    short = np.min([last_indx, len(no_cells_null[0])])
    ax.plot(time[0:short],no_cells_null[0][0:short], '--.', linewidth = 3,color='darkslateblue', label = str_null)
   
    short = np.min([last_indx, len(no_cells_44[0])])
    ax.plot(time[0:short],no_cells_44[0][0:short], '--.', linewidth = 3,color='royalblue', label = str_44 )
    
    short = np.min([last_indx, len(no_cells_33[0])])
    ax.plot(time[0:short],no_cells_33[0][0:short], '-.', linewidth = 3,color='forestgreen', label = str_33)
    
    short = np.min([last_indx, len(no_cells_22[0])])
    ax.plot(time[0:short],no_cells_22[0][0:short], ':', linewidth = 3,color='darkkhaki', label = str_22)
    
    short = np.min([last_indx, len(no_cells_11[0])])
    ax.plot(time[0:short], no_cells_11[0][0:short], '-', linewidth = 3,color='darkorange', label = str_11)
    
    short = np.min([last_indx, len(no_cells_00[0])])
    ax.plot(time[0:short], no_cells_00[0][0:short], '--', linewidth = 3,color='darkred', label = str_00)
    
    ax.set_xscale('log')
    ax.set_xlim(731,time[-1])
   
    
    lower_cell, upper_cell = custom_cell_interval
    ax.set_ylim(lower_cell,upper_cell)
    ax.set_xticks([731, 7301, 36501], [1, 10, 50], fontsize = 20)#ax[1].set_xticks([])

    #ax[no].set_xlabel('time (years)', fontsize = 22)

    #ax[1].set_xticks(list(range(0,40151,730*25)), list(range(0,55,25)), fontsize = 20)


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(2)
    
    ax.tick_params(width=2)
        

    fig.supxlabel('time [steps] (log$_{10}$)', y = -0.05)
    fig.supylabel('No. cells', x = -0.15)
    main_title_string = string_pdiv_dev0 + "\n " + 'Average no. cells'
    fig.suptitle(main_title_string, y=1.05)

    plt.legend(bbox_to_anchor=(1.1,0.9), borderaxespad=0,frameon=False, fontsize = 18) 
    save_it = os.path.join(output_directory, "zoom_no_cells.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    
    
    #%%
    
    

    no_cells_last = [ data_null_no_cells[last_indx_null-1][~np.isnan(data_null_no_cells[last_indx_null-1])],
                data_44_no_cells[last_indx_44-1][~np.isnan(data_44_no_cells[last_indx_44-1])],
                data_33_no_cells[last_indx_33-1][~np.isnan(data_33_no_cells[last_indx_33-1])],
                data_22_no_cells[last_indx_22-1][~np.isnan(data_22_no_cells[last_indx_22-1])],
                data_11_no_cells[last_indx_11-1][~np.isnan(data_11_no_cells[last_indx_11-1])],
                data_00_no_cells[last_indx_00-1][~np.isnan(data_00_no_cells[last_indx_00-1])]]

    fig, ax1 = plt.subplots(1,figsize=(5,5), sharex = True) # gridspec_kw={'width_ratios': [5, 4]}
    

    boxplt2 = ax1.boxplot(no_cells_last , showfliers=False , patch_artist=True) 



    for median in boxplt2['medians']:
        median.set_color('black')
        


    colors = list(reversed(['darkred', 'darkorange', 'darkkhaki', 'forestgreen', 'royalblue', 'darkslateblue']))


    for patch, color in zip(boxplt2['boxes'], colors):
        patch.set_facecolor(color)
        #patch.set_alpha(0.75)


   
    # # zoom-in / limit the view to different portions of the data
    ax1.set_ylim(lower_cell,upper_cell)  # outliers only
   

    ax1.set_yticks([lower_cell,upper_cell]) # , fontsize = 18
    
    ax1.set_xticks([1,2,3,4,5, 6], list(reversed(['1', '1e-1', '1e-2', '1e-3', '1e-4', '0'])))
    ax1.set_xlabel('probability of mutation $p_{mut}$', labelpad = 10) #, fontsize = 18



    #ax2.xaxis.tick_bottom()
    ax1.set_xlabel('$p_{mut}$', fontsize = 22) #, fontsize = 18
    fig.supylabel("w", fontsize = 22) #, fontsize = 18


    main_title_string = 'Average no cells at t = t$_{end}$ for individual simulations' #string_pdiv_dev0 + "\n " + 
    fig.suptitle(main_title_string, y=1.05)
    save_it = os.path.join(output_directory, "no_cells_boxplots_tend.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    
    #%% 
    ks_01,pks_01 = stats.ks_2samp(fit_last[5], fit_last[4])
    ks_02,pks_02 = stats.ks_2samp(fit_last[5], fit_last[3])
    ks_03,pks_03 = stats.ks_2samp(fit_last[5], fit_last[2])
    ks_04,pks_04 = stats.ks_2samp(fit_last[5], fit_last[1])
    ks_0n,pks_0n = stats.ks_2samp(fit_last[5], fit_last[0])
    ks_12,pks_12 = stats.ks_2samp(fit_last[4], fit_last[3])
    ks_13,pks_13 = stats.ks_2samp(fit_last[4], fit_last[2])
    ks_14,pks_14 = stats.ks_2samp(fit_last[4], fit_last[1])
    ks_1n,pks_1n = stats.ks_2samp(fit_last[4], fit_last[0])
    ks_23,pks_23 = stats.ks_2samp(fit_last[3], fit_last[2])
    ks_24,pks_24 = stats.ks_2samp(fit_last[3], fit_last[1])
    ks_2n,pks_2n = stats.ks_2samp(fit_last[3], fit_last[0])
    ks_34,pks_34 = stats.ks_2samp(fit_last[2], fit_last[1])
    ks_3n,pks_3n = stats.ks_2samp(fit_last[2], fit_last[0])
    ks_4n,pks_4n = stats.ks_2samp(fit_last[1], fit_last[0])
    
    
    text_ks = ['1:1e-1', '1:1e-2', '1:1e-3', '1:1e-4', '1:0', '1e-1:1e-2', '1e-1:1e-3', 
            '1e-1:1e-4', '1e-1:0', '1e-2:1e-3', '1e-2:1e-4', '1e-2:0', '1e-3:1e-4', '1e-3:0', '1e-4:0']
    
    all_ks = [ks_01, ks_02, ks_03, ks_04, ks_0n, ks_12, ks_13, 
            ks_14, ks_1n, ks_23, ks_24, ks_2n, ks_34, ks_3n, ks_4n]
    
    pks = [pks_01, pks_02, pks_03, pks_04, pks_0n, pks_12, pks_13, 
            pks_14, pks_1n, pks_23, pks_24, pks_2n, pks_34, pks_3n, pks_4n]
    
    sig_pks = list(np.asarray(pks) < 0.01) 
    
    
    df_ks_delta = pd.DataFrame(list(zip(text_ks, all_ks, pks, sig_pks)), columns =['pairs tested','ks2 stats', 'ks2 p-value', '1% significant?'])
    save_it = os.path.join(output_directory, "tend_fitness_ks_stats.csv")
    
    df_ks_delta.to_csv(save_it)
    
    #p_mut = 0
    fig, ax = plt.subplots(1, figsize = (5.5,5.5))

    for z in range(end_no_null):
        ax.plot(data_null_fit.loc[z], 'k-', linewidth = 1)
 
    ax.set_xlim(0, last_indx)
    ax.set_title('Individual simulation trajectories\n$p_{mut}=0$', y = 1.05)
    ax.set_xlabel('time step')
    ax.set_ylabel('fitness')
    save_it = os.path.join(output_directory, "null_fit_traj.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    #p_mut 1e-4
    fig, ax = plt.subplots(1, figsize = (5.5,5.5))
    for z in range(end_no_44):
        ax.plot(data_44_fit.loc[z], 'k-', linewidth = 1)
 
    ax.set_xlim(0, last_indx)
    ax.set_title('Individual simulation trajectories\n$p_{mut}=1e-4$', y = 1.05)
    ax.set_xlabel('time step')
    ax.set_ylabel('fitness')
    save_it = os.path.join(output_directory, "1e-4_fit_traj.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    #p_mut 1e-3
    fig, ax = plt.subplots(1, figsize = (5.5,5.5))
    for z in range(end_no_33):
        ax.plot(data_33_fit.loc[z], 'k-', linewidth = 1)
 
    ax.set_xlim(0, last_indx)
    ax.set_title('Individual simulation trajectories\n$p_{mut}=1e-3$', y = 1.05)
    ax.set_xlabel('time step')
    ax.set_ylabel('fitness')
    save_it = os.path.join(output_directory, "1e-3_fit_traj.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    
    #p_mut 1e-2
    fig, ax = plt.subplots(1, figsize = (5.5,5.5))
    for z in range(end_no_22):
        ax.plot(data_22_fit.loc[z], 'k-', linewidth = 1)
 
    ax.set_xlim(0, last_indx)
    ax.set_title('Individual simulation trajectories\n$p_{mut}=1e-2$', y = 1.05)
    ax.set_xlabel('time step')
    ax.set_ylabel('fitness')
    save_it = os.path.join(output_directory, "1e-2_fit_traj.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    #p_mut 1e-1
    fig, ax = plt.subplots(1, figsize = (5.5,5.5))
    for z in range(end_no_11):
        ax.plot(data_11_fit.loc[z], 'k-', linewidth = 1)
 
    ax.set_xlim(0, last_indx)
    ax.set_title('Individual simulation trajectories\n$p_{mut}=1e-1$', y = 1.05)
    ax.set_xlabel('time step')
    ax.set_ylabel('fitness')
    save_it = os.path.join(output_directory, "1e-1_fit_traj.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    #p_mut 1
    fig, ax = plt.subplots(1, figsize = (5.5,5.5))
    for z in range(end_no_00):
        ax.plot(data_00_fit.loc[z], 'k-', linewidth = 1)
 
    ax.set_xlim(0, last_indx)
    ax.set_title('Individual simulation trajectories\n$p_{mut}=1$', y = 1.05)
    ax.set_xlabel('time step')
    ax.set_ylabel('fitness')
    save_it = os.path.join(output_directory, "1_fit_traj.pdf")
    plt.savefig(save_it, bbox_inches='tight')
    
    
    plt.close('all')
  

    return "cat"
