# Synth_Data_Quality
Objective: create a novel metric to evaulate synthetic data quality for 2024 REU located at Lehigh University. 
Presentation: Graph Edit Metric (Lehigh REU) - Ethan Feldman.pdf

Data Sources (unzip the data_compressed.zip and put them in a "data" folder)
Adult: https://archive.ics.uci.edu/dataset/2/adult,
Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing,
Breast Cancer: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic,
Credit Card Approval: https://www.kaggle.com/datasets/samuelcortinhas/credit-card-approval-clean-data,
Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease,
Depression Dataset: https://www.kaggle.com/datasets/hamjashaikh/mental-health-detection-dataset,
Dermatology: https://archive.ics.uci.edu/dataset/33/dermatology,
Kidney Disease: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease,
CDC Diabetes: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset,
National Poll on Healthy Aging: https://archive.ics.uci.edu/dataset/936/national+poll+on+healthy+aging+(npha),
Student Performance: https://archive.ics.uci.edu/dataset/320/student+performance,
Titanic: https://www.kaggle.com/datasets/brendan45774/test-file

Idea:
- Original and Synthetic data are passed through a simple tree-based model
- The decision trees are collected and transformed into a graph based on the stem-leaf connections between features
- Graph edit distance between original and sythetic output graphs determine data utility

Outcome:
- GED and proposed GED score tends to agree with accuracy, f1, and roc auc scores. (eg. measurement of model on orginial data vs synthezied data are similar)
- Relatively useless for comparing data synethesizers.

Areas for Improvement:
- Synthesized data generators produce equally complex decision trees regardless of data synthesization method.
- This method does not scale well above 50 feature datasets.

Execution:
- Data synthesizer libraries may be out of date! 
