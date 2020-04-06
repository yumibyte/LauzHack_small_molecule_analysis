# Welcome to Small Molecule Analysis

These models are utilized to identify whether new drugs can be appropriately used for COVID-19

## IsDrugAcceptable Usage

'''
open the IsDrugAcceptable
'''

Set the working directory to the folder with the DataModified.csv file.This is the model and upon running it, the GUI should appear. All parameters that are necessary to fill are the molecular weight, TPSA, WLogp, Water Soluble, Bioavailability score, and number rot bonds. By clicking analyze, the model will process the data and output either "true" or "false".

## SMILESAnalysis Usage

'''
open eval.py

'''

Within this file set the working directory to the folder with the DataModified.csv file. Hit run and to see the amount of similarity between ATP and the given molecule, refer to "Gestalt Pattern Matching" in the created results.txt file.
