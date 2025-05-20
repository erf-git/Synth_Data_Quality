import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

PATH = "/home/erf6575/Desktop/SynthData/"
SEED = 42

#################### FILES ####################

def getFiles(name):
    
    files = [PATH + f"data_orig/{name}.csv",
            PATH + f"data_synth/ctgan_{name}.csv",
            PATH + f"data_synth/dpgan_{name}.csv",
            PATH + f"data_synth/gc_{name}.csv",
            PATH + f"data_synth/pate_{name}.csv",
            PATH + f"data_synth/pb_{name}.csv",
            PATH + f"data_synth/bn_{name}.csv"
            ]
    
    try:
        original = pd.read_csv(files[0])
    except:
        print("Orignal missing or misspelled ", files[0])
        original = None
    
    try:
        ctgan = pd.read_csv(files[1])
    except:
        print("CTGAN missing or misspelled ", files[1])
        ctgan = None
    
    try:
        dpgan = pd.read_csv(files[2])
    except:
        print("DPGAN missing or misspelled ", files[2])
        dpgan = None
    
    try:
        gc = pd.read_csv(files[3])
    except:
        print("Gaussian-Copula missing or misspelled ", files[3])
        gc = None
    
    try:
        pate = pd.read_csv(files[4])
    except:
        print("PATE-GAN missing or misspelled ", files[4])
        pate = None
    
    try:
        pb = pd.read_csv(files[5])
    except:
        print("PrivBayes missing or misspelled ", files[5])
        pb = None
    
    try:
        bn = pd.read_csv(files[6])
    except:
        print("Bayesian-Network missing or misspelled ", files[6])
        bn = None
    
    
    return original, ctgan, dpgan, gc, pate, pb, bn


def getScores(dataset, classifier, K=10):

    # Make sure all values are numeric
    ord_enc = OrdinalEncoder()
    for col in list(dataset.select_dtypes(include=['object']).columns):
        dataset[col] = ord_enc.fit_transform(dataset[[col]])

    # Define x and y
    x_input = dataset.drop(columns=['income']).to_numpy()
    y_target = dataset['income'].to_numpy()

    # Folds and result lists
    skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)
    acc_scores = []
    f1_scores = []
    auc_scores = []
    precision_scores = []
    recall_scores = []

    for i, (train_index, test_index) in enumerate(skf.split(x_input, y_target)):
        print(f"Fold {i+1}/{K}...")
        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")
        
        # Train the classifier
        classifier.fit(x_input[train_index], y_target[train_index])

        # Return predictions
        Y_pred = classifier.predict(x_input[test_index])
        Y_pred_prob = classifier.predict_proba(x_input[test_index])[:,1]

        # Evaluation
        acc = accuracy_score(y_target[test_index], Y_pred)
        # print("Accuracy", acc)
        f1 = f1_score(y_target[test_index], Y_pred, average='weighted')
        # print("F1:", f1)
        auc = roc_auc_score(y_target[test_index], Y_pred_prob, average='weighted')
        # print("AUC:", auc)
        precision = precision_score(y_target[test_index], Y_pred, average='weighted')
        # print("Precision:", precision)
        recall = recall_score(y_target[test_index], Y_pred, average='weighted')
        # print("Recall:", recall)
        
        acc_scores.append(acc)
        f1_scores.append(f1)
        auc_scores.append(auc)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    scores = {
        'accuracy': acc_scores,
        'f1': f1_scores,
        'auc': auc_scores,
        'precision': precision_scores,
        'recall': recall_scores
    }
    
    return scores


def getResults(name, classifier):
    
    original, ctgan, dpgan, gc, pate, pb, bn = getFiles(name)
    
    # None of the sklearn models work with na data
    original, ctgan, dpgan, gc, pate, pb, bn = original.dropna(), ctgan.dropna(), dpgan.dropna(), gc.dropna(), pate.dropna(), pb.dropna(), bn.dropna()
    
    original_scores = getScores(original, classifier)
    ctgan_scores = getScores(ctgan, classifier)
    dpgan_scores = getScores(dpgan, classifier)
    gc_scores = getScores(gc, classifier)
    pate_scores = getScores(pate, classifier)
    pb_scores = getScores(pb, classifier)
    bn_scores = getScores(bn, classifier)
    
    results = {
        'original': original_scores,
        'ctgan': ctgan_scores,
        'dpgan': dpgan_scores,
        'gc': gc_scores,
        'pate': pate_scores,
        'pb': pb_scores,
        'bn': bn_scores,
    }
    
    return results



adult_rfc = getResults("adult", RandomForestClassifier(class_weight="balanced", random_state=SEED))



o_df = pd.DataFrame(adult_rfc['original'])
c_df = pd.DataFrame(adult_rfc['ctgan'])
d_df = pd.DataFrame(adult_rfc['dpgan'])
pa_df = pd.DataFrame(adult_rfc['pate'])
g_df = pd.DataFrame(adult_rfc['gc'])
p_df = pd.DataFrame(adult_rfc['pb'])
b_df = pd.DataFrame(adult_rfc['bn'])

o_df.describe()
c_df.describe()
d_df.describe()


# # None of the sklearn models work on 
# original, ctgan, dpgan, gc, pate, pb, bn = getFiles("adult")
# original, ctgan, dpgan, gc, pate, pb, bn = original.dropna(), ctgan.dropna(), dpgan.dropna(), gc.dropna(), pate.dropna(), pb.dropna(), bn.dropna()

# o = getScores(original, RandomForestClassifier(class_weight="balanced", random_state=SEED))
# c = getScores(ctgan, RandomForestClassifier(class_weight="balanced", random_state=SEED))
# d = getScores(dpgan, RandomForestClassifier(class_weight="balanced", random_state=SEED))
# pa = getScores(pate, RandomForestClassifier(class_weight="balanced", random_state=SEED))
# g = getScores(gc, RandomForestClassifier(class_weight="balanced", random_state=SEED))
# p = getScores(pb, RandomForestClassifier(class_weight="balanced", random_state=SEED))
# b = getScores(bn, RandomForestClassifier(class_weight="balanced", random_state=SEED))

# o_df = pd.DataFrame(o)
# c_df = pd.DataFrame(c)
# d_df = pd.DataFrame(d)
# pa_df = pd.DataFrame(pa)
# g_df = pd.DataFrame(g)
# p_df = pd.DataFrame(p)
# b_df = pd.DataFrame(b)