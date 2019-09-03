# Random Selection 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the Ads Click Through Rate Dataset 
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection, where we iterate through each user and grab the 0/1 value for the specific ad selected
import random
N = 10000 # Number of recipients
d = 10 # Number of ads
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
    
# Visualising the results - Histogram
plt.hist(ads_selected, ec = 'black')
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

