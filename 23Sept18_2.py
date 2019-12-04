# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 11:35:48 2018

@author: srahman
"""
%reset

import os
os.chdir('C:/Users/srahman/Google Drive/Bond_Pricing/Trace_Data_Sample')

import numpy as np
import pandas as pd

td = pd.read_csv('CEP_20160701.csv', header=None, skiprows = 0, skipinitialspace=True, usecols=([15,21,24,64]))

td = td[pd.notnull(td[64])]
td = td[td[64] != 0]

mid = td[15]
mid = pd.to_numeric(mid,'coerce')
mid = mid.dropna()

bid = td[21]
bid = pd.to_numeric(bid,'coerce')
bid = bid.dropna()

ask = td[24]
ask = pd.to_numeric(ask,'coerce')
ask = ask.dropna()

trade = td[64]
trade = pd.to_numeric(trade,'coerce')
trade = trade.dropna()


spreadsandwich = np.empty(np.count_nonzero(trade))
spreadsandwich = trade
spreadsandwich = pd.to_numeric(spreadsandwich,'coerce')
spreadsandwich = np.where(spreadsandwich <= ask, spreadsandwich, 'nan')
spreadsandwich = pd.to_numeric(spreadsandwich,'coerce')

spreadsandwich = np.where(spreadsandwich >= bid, spreadsandwich, 'nan')

spreadsandwich = pd.to_numeric(spreadsandwich,'coerce')
btwthebread = (np.count_nonzero(~np.isnan(spreadsandwich))/np.count_nonzero(trade))*100
np.savetxt('C:/Users/srahman/Google Drive/Bond_Pricing/Between_The_Spread/20160701.csv',spreadsandwich)

delta = trade - mid
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')
dist_plot_raw_delta = sns.distplot(delta)
plt.savefig("dist_plot_raw_delta.png")
delta = 

# td.loc[td[trade] <= td[ask] & td[trade] >= td[bid] , 'SPREAD_SANDWICH'] = 1
# td.loc[td[trade] > td[ask] & td_trade[trade] < td_trade[bid] , 'SPREAD_SANDWICH'] = 0


