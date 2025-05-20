import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

PATH = "/home/erf6575/Desktop/SynthData/"


#################### EDA ####################

files = {
    "adult": PATH + "data_orig/adult.csv",
    "bank": PATH + "data_orig/bank.csv",
    "breast_cancer" :PATH + "data_orig/breast_cancer.csv",
    "card_approval": PATH + "data_orig/card_approval.csv",
    "cleveland_heart": PATH + "data_orig/cleveland_heart.csv",
    "depression": PATH + "data_orig/depression.csv",
    "dermatology": PATH + "data_orig/dermatology.csv",
    "diabetes": PATH + "data_orig/diabetes.csv",
    "kidney_disease" :PATH + "data_orig/kidney_disease.csv",
    "NPHA_doctor_visits": PATH + "data_orig/NPHA_doctor_visits.csv",
    "student_math": PATH + "data_orig/student_math.csv",
    "titanic": PATH + "data_orig/titanic.csv"
}

for name, file in files.items():
    
    print(name)
    dataset = pd.read_csv(file)
    print(dataset.nunique())
    
    print()


# adult: 42
# bank: 15
# breast_cancer: 10
# card_approval: 15
# cleveland_heart: 10
# depression: 15
# dermatology: 10
# diabetes: 10
# kidney_disease: 5
# NPHA_doctor_visits: 5
# student_math: 10
# titanic: 10

# I changed some boolean values to 0 and 1
# yes: 1, no: 0
# normal: 0, abnormal: 1
# M: 1, B: 0
# 
# ...
# It doesn't matter in the greater scheme which gets assigned 1 or 0. 
# The important thing is that column remains with 2 values total in relation to the original study.
# Setting boolean values to 0 and 1 make is easier when setting up DataSynthesizer