# Apriori

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the Market Basket Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range (0,20)])
    
# Training Apriori on the Dataset
from apyori import apriori
# Get the min_support by [(# item is purchased per day * 7) / total # of transactions]
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualising the results
results = list(rules)

#Generating new list formats so that we can read the results
results_list = []
for i in range(0, len(results)):
    results_list.append('\RESULTS:\t' + str(results[i][2]))