import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
import networkx as nx
from TreeHelp import Node
from TreeScores import build_zss_tree, tree_to_digraph, print_tree, getScores

# PATH = "/home/erf6575/Desktop/SynthData/"
PATH = os.getcwd() + "/"
print(PATH)


# Creates a dataset of desired shape of random distribution
class Dataset:
    
    def __init__(self, name, np_seed, sklearn_seed, rows, columns, correct_y=True):
        
        self.name = name
        
        # Numpy random
        np.random.seed(seed=np_seed)
        self.x = np.random.rand(rows, columns)
        if correct_y:
            self.y = np.round(np.mean(self.x, axis=1))
        else:
            self.y = np.round(np.random.rand(rows, 1))
        
        self.n_obs = self.x.shape[0]
        
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, 
                                                            test_size=0.20, stratify=self.y, 
                                                            random_state=sklearn_seed)

        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def __repr__(self):
        return f"Dataset <x_train: {self.x_train.shape}, x_test: {self.x_test.shape}, y_train: {self.y_train.shape}, y_test: {self.y_test.shape}>"
        
    def __str__(self):
        return self.__repr__()
        

# Of 
def getTrees(timeout, save_path):

    # Data
    # df_a: original dataset
    # df_b: similar dataset (similar y values, but different seed)
    # df_c: dissimilar dataset (random y values)
    df_a = Dataset(name="A", np_seed=42, sklearn_seed=42, rows=50, columns=3)
    df_b = Dataset(name="B", np_seed=42, sklearn_seed=0, rows=50, columns=3)
    df_c = Dataset(name="C", np_seed=42, sklearn_seed=42, rows=50, columns=3, correct_y=False)
    
    
    # A
    tree_a = DecisionTreeClassifier(random_state=42)
    tree_a.fit(df_a.x_train, df_a.y_train)
    tree.plot_tree(tree_a, fontsize=6)
    plt.savefig(PATH + 'figures/tree_a.png')
    
    # B
    tree_b = DecisionTreeClassifier(random_state=42)
    tree_b.fit(df_b.x_train, df_b.y_train)
    tree.plot_tree(tree_b, fontsize=6)
    plt.savefig(PATH + 'figures/tree_b.png')
    
    # C
    tree_c = DecisionTreeClassifier(random_state=42)
    tree_c.fit(df_c.x_train, df_c.y_train)
    tree.plot_tree(tree_c, fontsize=6)
    plt.savefig(PATH + 'figures/tree_c.png')
    
    
    # Convert classifier tree to graph
    tree_original = build_zss_tree(tree_b.tree_)
    graph_original = tree_to_digraph(tree_original, df_b.n_obs)
    print("original tree and graph built")
    
    # Compute accuracy measures
    scores_df = pd.DataFrame(columns=['graph', 'accuracy', 'f1', 'auc', 'ged', 'ged_score'])
    
    scores_df = getScores(tree_b, df_b, graph_original, scores_df, timeout)
    scores_df = getScores(tree_a, df_a, graph_original, scores_df, timeout)
    scores_df = getScores(tree_c, df_c, graph_original, scores_df, timeout)
    
    scores_df.to_csv(save_path, index=False)


# Run example trees and metrics
getTrees(300, PATH+"scores/example_trees.csv")





