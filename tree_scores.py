import os
import pandas as pd 
import numpy as np 
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn import tree
import networkx as nx
from TreeHelp import Node

# PATH = "/home/erf6575/Desktop/SynthData/"
PATH = os.getcwd() + "/"
SEED = 42
print(PATH)

class Dataset:
    
    def __init__(self, filePath, name):
        
        self.name = name
        
        # Read file
        dataset = pd.read_csv(filePath).dropna()
        
        self.n_obs = dataset.shape[0]

        # Make sure all values are numeric
        encoder = OrdinalEncoder()
        for col in list(dataset.select_dtypes(include=['object']).columns):
            dataset[col] = encoder.fit_transform(dataset[[col]])

        # Define x and y
        x_input = dataset.drop(columns=['y'])
        y_target = dataset['y']

        X_train, X_test, y_train, y_test = train_test_split(x_input.to_numpy(), y_target.to_numpy(), test_size=0.20, random_state=42, stratify=y_target)
    
        self.x_train = X_train
        self.x_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.x_input = x_input
        self.y_target = y_target
        
    def __repr__(self):
        return f"Dataset <x_train: {self.x_train.shape}, x_test: {self.x_test.shape}, y_train: {self.y_train.shape}, y_test: {self.y_test.shape}>"
        
    def __str__(self):
        return self.__repr__()
        
def print_tree(node, depth=0):
    if node is None:
        return
    # Print the current node with indentation based on its depth
    print(" " * depth * 2 + f"Node(label={node.label}, value={node.value})")
    # Recursively print each child
    for child in node.children:
        print_tree(child, depth + 1)

def build_zss_tree(tree, node_id=0):
    """Helper function to construct tree."""
    if node_id == -1:
        return None
    
    # Check if a leave node
    if tree.feature[node_id] == -2:
        class_label = np.argmax(tree.value[node_id][0])
        node = Node(label=f"Class_{class_label}", value=tree.value[node_id])
    else:
        node = Node(label=tree.feature[node_id], value=tree.value[node_id])
    
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    
    if left_child != -1:
        node.addkid(build_zss_tree(tree, left_child))
    if right_child != -1:
        node.addkid(build_zss_tree(tree, right_child))
    
    return node

def add_nodes_edges(graph, node, parent=None):
    """Helper function to add nodes and edges to the graph."""
    
    # Add or update the node
    if not graph.has_node(node.label):
        graph.add_node(node.label, label=node.label, value=node.value)
    else:
        # Static average
        current = graph.nodes[node.label]['value']
        new = node.value
        graph.nodes[node.label]['value'] = np.concatenate((current, new), axis=0)
    
    # Add or update the edge if parent
    if parent is not None:
        if graph.has_edge(parent.label, node.label):
            current = graph[parent.label][node.label]['weight']
            new =  np.array([1]) #np.abs(np.sum(parent.value[0]) - np.sum(node.value[0]))
            graph[parent.label][node.label]['weight'] = np.append(current, new)
        else:
            current = np.array([1]) #np.abs(np.sum(parent.value[0]) - np.sum(node.value[0])) 
            graph.add_edge(parent.label, node.label, path=f"[{parent.label}][{node.label}]", weight=current)
            
        
    # Recurse  
    for child in node.children:
        add_nodes_edges(graph, child, node)

def tree_to_digraph(root, n_obs):
    """Converts a tree to a NetworkX directed graph."""
    G = nx.DiGraph()
    add_nodes_edges(G, root)
    
    # Caculates average at the end
    for node in G.nodes:
        G.nodes[node]['value'] = np.expand_dims(np.mean(G.nodes[node]['value'], axis=0), axis=0)
    for u, v, data in G.edges(data=True):
        G[u][v]['weight'] = np.mean(G[u][v]['weight']) / n_obs
        
    return G

from sklearn.metrics.pairwise import cosine_similarity

def node_subst_cost(u, v):
    
    # Check if same name
    if u['label'] == v['label']:
        
        # Check if same weight
        if np.array_equal(u['value'][0], v['value'][0]):
            return 0
        else:
            # If similar then cosine_sim == 1, but we want it to be similar, so it's the inverse
            # print(u, v)
            return 1 - cosine_similarity(u['value'], v['value'])
    
    else:
        return 1

def edge_subst_cost(u, v):
    
    # Check if same path
    if u['path'] == v['path']:
        
        # Check if same weight
        if np.array_equal(u['weight'], v['weight']):
            return 0
        else:
            # These weights give the percentage of observations that on average that go down tis path
            # It is between 0-1, so subtracting them will also give between 0-1
            # print(u, v)
            # return np.abs(u['weight'] - v['weight']) 
            return 0.5
    
    else:
        return 1


from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

# Assuming all costs are between 0-1...
# The maximum number of operations required to transform one tree into another is determined by the structure and labels of the trees. 
# In the worst-case scenario, you would need to remove all nodes from one tree and then insert all nodes from the other tree. 
# This happens when the two trees have completely different structures and labels.
def count_graph(G):
    return len(G.nodes) + len(G.edges)

def count_between(graphA, graphB):
    return count_graph(graphA) + count_graph(graphB)


def getScores(classifier, data, graph_original, df, timeout=1200):
    
    # Name
    name = data.name
    
    # Return predictions
    Y_pred = classifier.predict(data.x_test)
    
    # Evaluation
    acc = accuracy_score(data.y_test, Y_pred)
    # print("Accuracy", acc)
    f1 = f1_score(data.y_test, Y_pred, average='macro')
    # print("F1:", f1)
    # precision = precision_score(data.y_test, Y_pred)
    # # print("Precision:", precision)
    # recall = recall_score(data.y_test, Y_pred)
    # # print("Recall:", recall)
    
    if len(classifier.classes_) == 2:
        Y_pred_prob = classifier.predict_proba(data.x_test)[:, 1]
        auc = roc_auc_score(data.y_test, Y_pred_prob)
    elif len(classifier.classes_) > 2:
        Y_pred_prob = classifier.predict_proba(data.x_test)
        auc = roc_auc_score(data.y_test, Y_pred_prob, multi_class='ovo')
    else:
        print("what?")
    # print("AUC:", auc)
    
    
    # Convert classifier tree to graph
    tree = build_zss_tree(classifier.tree_)
    graph = tree_to_digraph(tree, data.n_obs)
    
    # Inverse of all 
    ged = nx.graph_edit_distance(graph_original, graph, node_subst_cost=node_subst_cost, edge_subst_cost=edge_subst_cost, timeout=timeout)
    ged_score = 1 - ( ged / count_between(graph_original, graph) )
    
    
    # df.loc[len(df.index)] = [name, acc, f1, auc, precision, recall, ged] 
    new = {'synthesizer': name, 'accuracy': acc, 'f1': f1, 'auc': auc, 'ged': ged, 'ged_score': ged_score}
    df = df._append(new, ignore_index = True)
    print(df)
    
    return df


from sklearn.tree import DecisionTreeClassifier

def getMetrics(name, timeout, save_path):

    # Data
    df_o = Dataset(PATH + f"data/{name}_original.csv", "original")
    df_g = Dataset(PATH + f"data/{name}_gc.csv", "GaussianCopula")
    df_c = Dataset(PATH + f"data/{name}_ctgan.csv", "CTGAN")
    df_p = Dataset(PATH + f"data/{name}_pate.csv", "PATEGAN")
    df_b = Dataset(PATH + f"data/{name}_pb.csv", "PrivBayes")
    

    clf_original = DecisionTreeClassifier(random_state=42)
    clf_original.fit(df_o.x_train, df_o.y_train)
    # Convert classifier tree to graph
    tree_original = build_zss_tree(clf_original.tree_)
    graph_original = tree_to_digraph(tree_original, df_o.n_obs)
    print("original tree and graph built")

    clf_gc = DecisionTreeClassifier(random_state=42)
    clf_gc.fit(df_g.x_train, df_g.y_train)
    print("gc tree and graph built")

    clf_ctgan = DecisionTreeClassifier(random_state=42)
    clf_ctgan.fit(df_c.x_train, df_c.y_train)
    print("ctgan tree and graph built")

    clf_pate = DecisionTreeClassifier(random_state=42)
    clf_pate.fit(df_p.x_train, df_p.y_train)
    print("pate tree and graph built")

    clf_pb = DecisionTreeClassifier(random_state=42)
    clf_pb.fit(df_b.x_train, df_b.y_train)
    print("pb tree and graph built")
    
    
    scores_df = pd.DataFrame(columns=['synthesizer', 'accuracy', 'f1', 'auc', 'ged', 'ged_score'])

    scores_df = getScores(clf_original, df_o, graph_original, scores_df, timeout)
    scores_df = getScores(clf_gc, df_g, graph_original, scores_df, timeout)
    scores_df = getScores(clf_ctgan, df_c, graph_original, scores_df, timeout)
    scores_df = getScores(clf_pate, df_p, graph_original, scores_df, timeout)
    scores_df = getScores(clf_pb, df_b, graph_original, scores_df, timeout)
    
    scores_df.to_csv(save_path, index=False)
    # return scores_df
    

# getMetrics("adult")
# getMetrics("bank")
# getMetrics("cancer")
# getMetrics("card")
# getMetrics("dermatology")
# getMetrics("diabetes")
# getMetrics("heart")
# getMetrics("iris")
# getMetrics("titanic")


getMetrics("cancer", 1800, PATH + f"data/cancer_scores_30.csv")
getMetrics("cancer", 900, PATH + f"data/cancer_scores_15.csv")
getMetrics("cancer", 300, PATH + f"data/cancer_scores_5.csv")
getMetrics("cancer", 60, PATH + f"data/cancer_scores_1.csv")



