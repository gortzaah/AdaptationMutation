#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:29:14 2024

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
from functions_read_averages import *

from matplotlib.ticker import ScalarFormatter, NullFormatter
from scipy import stats
plt.rcParams.update({'font.size': 20})


#where the data ara located

### STRING TO TEST 
#main_string = "/Users/80021045/Dropbox/PhD project/HAL-Results/adaptive_pop_May2024/"

i = 0 #which pdiv

bounds = ["05_bounds", "50_bounds", "33_bounds", "10_bounds",
          "big_skew_0neutral_bounds","big_skew_40neutral_bounds",
          "small_skew_0neutral_bounds","small_skew_40neutral_bounds"]


for i in range(5):
    for bound in bounds:
        main_string = "/Users/80021045/Dropbox/PhD project/HAL-Results/gosia_june/" + bound + "/std=0.000/"
        
        pdiv_string_array = ["pdiv0=0.100", "pdiv0=0.250", "pdiv0=0.350", "pdiv0=0.700", "pdiv0=1.000"]
        pdiv_double_array = [0.1, 0.25, 0.35, 0.7, 1]
        mutation_string_array = ["both_0", "both_1e-4",
                                  "both_1e-3", "both_1e-2", "both_1e-1", "both_1"]
        
        
        pdiv = pdiv_string_array[i]
        pdiv_main_string = main_string + pdiv
    
        
        # #where to output files
        destination_string = '/Users/80021045/Dropbox/PhD project/MP_code/experiments_june2024/graphs/' + bound
        print("Path exists? " + str(os.path.exists(destination_string)))
        output_directory = os.path.join(destination_string, pdiv_string_array[i])
        print("Path exists? " + str(os.path.exists(output_directory)))
        print(pdiv_string_array[i] )
        print(bound)
        
        
        custom_cell_interval = [9000,9700]
        custom_fitness_interval = [0.7,0.81]
        main_executive_function(pdiv_main_string, mutation_string_array,
                                pdiv_double_array[i], output_directory, custom_cell_interval, custom_fitness_interval)
        


#%%
# open_it = os.path.join(output_directory, "stats_last.pkl")
# test_pickled_data = pd.read_pickle(open_it) 











