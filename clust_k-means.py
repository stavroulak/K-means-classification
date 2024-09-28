# If you need:
# !pip install scikit-learn==0.23.1

# Unsupervised k-means classification

import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score, f1_score
from sklearn.metrics import classification_report

# Input as a csv file (all the classified data to train your model):
input = 'morphology.csv'
# The names of your columns which will be used to classify tha data
columns_x = ['log(W1)', 'log(NUV)', 'n'] # , 'WISE_3.4' , 'SPIRE_250', 'GALEX_NUV', 'q',
# The name of your column with the classification (the categories to me numerical integers: e.g. 0 and 1)


# Main

df = pd.read_csv(input)
df = df.dropna(axis=0)
df['log(W1)'] = np.log(df['WISE_3.4'].values)
df['log(NUV)'] = np.log(df['GALEX_NUV'].values)
df['log(250)'] = np.log(df['SPIRE_250'].values)
X = df[columns_x].values
X = np.nan_to_num(X)

# Normalize the data:
Clus_dataSet = StandardScaler().fit_transform(X)

clusterNum = 2 # number of the clusters
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12) # n_init: number of the iterations
k_means.fit(X)
labels = k_means.labels_ # the labels the data have been received

df["Clus_km"] = labels

print("f1-score:")
print(f1_score(df.type, df.Clus_km, average='weighted'))

print("Jaccard score:")
print(jaccard_score(df.type, df.Clus_km, pos_label=0))

print("Classification report:")
print(classification_report(df.type, df.Clus_km))

df.to_csv("clustering.csv")

print(df.groupby('Clus_km').mean()) # check the centroid values

# Make a 2D plot
area = np.pi * ( X[:, 1])**2
plt.scatter(X[:, 0], X[:, 2], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Label_x', fontsize=18)
plt.ylabel('Label_y', fontsize=16)
plt.savefig("clustering2D.png")


# Make a 3D plot
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.set_xlabel('Label_x')
ax.set_ylabel('Label_y')
ax.set_zlabel('Label_z')

ax.scatter(X[:, 1], X[:, 0], X[:, 2], c= labels.astype(np.float))
plt.savefig("clustering3D.png")
