#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:42:20 2020

@author: ashleyraigosa
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_excel('Data.xlsx')
X = dataset.iloc[:, 2:9].values
y = dataset.iloc[:, 9].values