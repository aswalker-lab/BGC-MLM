# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:57:50 2026

@author: Allison Walker
Butina clustering adapted from https://projects.volkamerlab.org/teachopencadd/talktorials/T005_compound_clustering.html
"""
import argparse
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import HDBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from rdkit.ML.Cluster import Butina

def replace_outliers_with_singletons(labels, outlier_label=-1):
    """
    Replace outlier labels, usually -1 from OPTICS/HDBSCAN, with unique singleton
    cluster labels.

    Parameters
    ----------
    labels : array-like
        Cluster labels from OPTICS, HDBSCAN, DBSCAN, etc.
    outlier_label : int, default=-1
        Label used to mark outliers/noise.

    Returns
    -------
    new_labels : np.ndarray
        Cluster labels where each outlier has been assigned its own unique cluster.
    """
    labels = np.asarray(labels).copy()

    outlier_mask = labels == outlier_label
    n_outliers = np.sum(outlier_mask)

    if n_outliers == 0:
        return labels

    non_outlier_labels = labels[~outlier_mask]

    if len(non_outlier_labels) == 0:
        next_cluster = 0
    else:
        next_cluster = non_outlier_labels.max() + 1

    labels[outlier_mask] = np.arange(
        next_cluster,
        next_cluster + n_outliers
    )

    return labels

parser = argparse.ArgumentParser()            
parser.add_argument("similarity_file")
parser.add_argument("output_dir")
args = parser.parse_args()

sim_file_base = args.similarity_file[args.similarity_file.rfind("/"):args.similarity_file.rfind(".")]

similarity_dic = {}
bgc_list = []
for line in open(args.similarity_file):
    split_line = line.split(",")
    bgc1 = split_line[0]
    bgc2 = split_line[1]
    if bgc1 not in similarity_dic:
        similarity_dic[bgc1] = {}
        bgc_list.append(bgc1)
    if bgc2 not in similarity_dic:
        similarity_dic[bgc2] = {}
        bgc_list.append(bgc2)
   
    sim = float(split_line[2])
    similarity_dic[bgc1][bgc2] = sim
    similarity_dic[bgc2][bgc1] = sim
    
for bgc in bgc_list:
    similarity_dic[bgc][bgc] = 1
    
similarity_matrix = np.zeros((len(bgc_list),len(bgc_list)))

for i in range(0,len(bgc_list)):
    bgc1 = bgc_list[i]
    for j in range(0, len(bgc_list)):
        bgc2 = bgc_list[j]
        similarity_matrix[i,j] = similarity_dic[bgc1][bgc2]
distance_matrix = 1 - similarity_matrix
np.fill_diagonal(distance_matrix, 0)

Z = linkage(squareform(distance_matrix), method="average")

# Example: cluster together points with distance <= 0.3
# Equivalent to similarity >= 0.7 if D = 1 - S
hierarchical_labels = fcluster(Z, t=0.2, criterion="distance")
#print(hierarchical_labels)
outfile = open(args.output_dir + "/" + sim_file_base + "_hierarchical.csv",'w')
for i in range(0,len(bgc_list)):
    outfile.write(bgc_list[i] + "," + str(hierarchical_labels[i]) + "\n")
outfile.close()

clusterer = HDBSCAN(
    metric="precomputed",
    min_cluster_size=2
)

hdbscan_labels = clusterer.fit_predict(distance_matrix)
hdbscan_labels = replace_outliers_with_singletons(hdbscan_labels)
outfile = open(args.output_dir + "/" + sim_file_base + "_hdbscan.csv",'w')
for i in range(0,len(bgc_list)):
    outfile.write(bgc_list[i] + "," + str(hdbscan_labels[i]) + "\n")
outfile.close()

model = OPTICS(
    metric="precomputed",
    min_samples=5,
    xi=0.05,
    min_cluster_size=2
)

optics_labels = model.fit_predict(distance_matrix)
optics_labels = replace_outliers_with_singletons(optics_labels)
outfile = open(args.output_dir + "/" + sim_file_base + "_optics.csv",'w')
for i in range(0,len(bgc_list)):
    outfile.write(bgc_list[i] + "," + str(optics_labels[i]) + "\n")
outfile.close()

model = AffinityPropagation(
    affinity="precomputed",
    random_state=0
)

ap_labels = model.fit_predict(similarity_matrix)
outfile = open(args.output_dir + "/" + sim_file_base + "_affinity_prop.csv",'w')
for i in range(0,len(bgc_list)):
    outfile.write(bgc_list[i] + "," + str(ap_labels[i]) + "\n")
outfile.close()

triangular_dist_matrix = []
for i in range(0, len(distance_matrix)):
    for j in range(i):
        triangular_dist_matrix.append(distance_matrix[i][j])
triangular_dist_matrix = np.array(triangular_dist_matrix)

clusters = Butina.ClusterData(triangular_dist_matrix, len(bgc_list), 0.2, isDistData=True)
clusters = sorted(clusters, key=len, reverse=True)
outfile = open(args.output_dir + "/" + sim_file_base + "_butina.csv",'w')
i = 0
j = 0
for c in clusters:
    for bgc in c:
        outfile.write(bgc_list[i] + "," + str(j) + "\n")
        i += 1
    j += 1
outfile.close()
    
