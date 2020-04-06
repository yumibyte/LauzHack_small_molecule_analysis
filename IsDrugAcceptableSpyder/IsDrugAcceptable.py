#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 10:42:20 2020
@author: ashleyraigosa
"""

    
# Creating GUI
import tkinter as tk
from tkinter.filedialog import askopenfilename
import pandas as pd

class Application(tk.Frame):
    

        
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.grid()
        self.create_widgets()
    
    def create_widgets(self):
        self.topiclbl = tk.Label(self, text='Import .csv file with drug data', font=(12))
        self.topiclbl.grid()
        self.browseButton_CSV = tk.Button(self,text="      Import CSV File     ", command=self.getCSV, bg='green', fg='white', font=('helvetica', 12, 'bold'))
        self.browseButton_CSV.grid()
        
        self.topiclbl = tk.Label(self, text='Analyze .csv', font=(12))
        self.topiclbl.grid()
        self.browseButton_CSV = tk.Button(self,text="      Analyze     ", command=self.MLModel, bg='green', fg='white', font=('helvetica', 12, 'bold'))
        self.browseButton_CSV.grid()
        
    def getCSV(v):
    
        csv_file_path = askopenfilename()
        return csv_file_path
    
    def MLModel(v):
        # Importing the libraries for ML
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        
        dataset = pd.read_csv(v)
        X = dataset.iloc[:, 2:9].values
        y = dataset.iloc[:, 9].values
        
        # Encoding categorical data
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        labelencoder_X_1 = LabelEncoder()
        X[:, 4] = labelencoder_X_1.fit_transform(X[:, 4])
        onehotencoder = ColumnTransformer([('one_hot_encoder',OneHotEncoder(),[4])],remainder='passthrough')
        X = onehotencoder.fit_transform(X)
        X = X[:, 1:]
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        # Create the ANN
        # Import Keras librariers and packages
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        
        # Initializing ANN
        classifier = Sequential()
        
        # Adding input layer and first hidden layer
        classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu', input_dim = 7))
        
        # Adding the second hidden layer
        classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))
        
        # Adding output layer
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
        
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # Fitting ANN to Training set
        classifier.fit(X_train, y_train, batch_size = 5, nb_epoch = 4)
        
        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        
        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

root = tk.Tk()
app = Application(master=root)
root.title("LauzHack Project")
canvas1 = tk.Canvas(root, width=300, height=300, bg='lightsteelblue2', relief='raised')
root.geometry('600x600')
v = tk.StringVar()


root.mainloop()


"""
# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer): 
    classifier = Sequential()
    classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 8))
    classifier.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [2, 4],
              'nb_epoch': [4, ],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_
"""
