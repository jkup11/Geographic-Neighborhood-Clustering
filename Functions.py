
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import sklearn
from operator import add
import csv 
import gmaps
import overpy
import geoplotlib
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import warnings
import geopandas
from geopandas.tools import sjoin
from matplotlib.patches import Polygon
from shapely.geometry import shape, Point
import shapely
import fiona
from copy import deepcopy
from math import log, radians, cos, sin, asin, sqrt
from decimal import Decimal
import time
import geopy.distance
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import math

import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.cluster import KMeans

# note: learn more about EMD and how to use it
#from pyemd import emd
from pyemd import emd_samples


# function that takes in data and the column where it finds cluster labels
# and returns the sizes of each 
def sd(data, cluster_col):
    e_data = data[data[cluster_col] > -1]
    max_clust = np.max(list(e_data[cluster_col]))
    sd_list = []
                       
    for i in list(set(list(e_data[cluster_col]))):
        clust_data = e_data[e_data[cluster_col] == i]
        sd = math.sqrt(np.var(list(clust_data['All Crime'])))
        sd_list.append(sd)
        
    return sd_list

def emd(data, cluster_col):
    clusters = data[cluster_col]
    max_clust = np.max(list(clusters))
    
    cluster_emd_vecs = []
    # iterate through clusters
    for cluster in range(max_clust+1):
        cluster_data = data[data[cluster_col] == cluster]
        cluster_size = len(cluster_data)
        if cluster_size == 0:
            #print("Cluster Size 0 error")
            continue
        
        crime_list = list(cluster_data['All Crime'])
        neighborhood_mean = np.mean(crime_list)
        sq_mu = math.sqrt(neighborhood_mean)
        
        sample_emd = []
        # average to eliminate impact of randomness on calculation
        # make result more consistent
        for j in range(50):
            comparison_dist = []
            for i in range(len(crime_list)):
                comparison_dist.append(neighborhood_mean + np.random.uniform(-1.0*sq_mu, sq_mu))
            sample_emd.append(emd_samples(crime_list, comparison_dist))
        cluster_emd_vecs.append(np.mean(sample_emd))

    return cluster_emd_vecs

def hav_dist(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    c = 2 * asin(sqrt(sin((lat2 - lat1) /2)**2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) /2)**2)) * 6371 # Radius of earth in kilometers
    return c

def pure_geo_k_means(data, k, cap=1000, verbose=False):
    # store error for iteration analysis
    s1 = time.time()
    error_mags = []
    sse = []
    C_x = []
    C_y = []
    rand_inds = random.sample(range(0, len(data)), k)

    arr = data.iloc[rand_inds][['Latitude','Longitude']]
    centroids_curr = np.array(arr, dtype=np.float32)
    # centroid storage
    centroids_old = np.zeros((k,2))
    # cluster label
    clusters = np.zeros(len(data))
    ideal_error = list(np.zeros(k))
    error = []
    
    for i in range(k):
        # lats and lons for haversine distance
        error.append(hav_dist(centroids_old[i][0], centroids_old[i][1], centroids_curr[i][0], centroids_curr[i][1]))
    itera = 0
    while not error == ideal_error:
        s2 = time.time()
        itera += 1
        
        # stop early
        if itera > cap:
            break
        
        if verbose:
            print("Iteration: " + str(itera))
        
        for i in range(len(data)):
            distances = []
            lat = data.at[i, 'Latitude']
            lon = data.at[i, 'Longitude']
            for j in range(len(centroids_curr)):
                distances.append(hav_dist(centroids_curr[j][0], centroids_curr[j][1], lat, lon))
            cluster_num = np.argmin(distances)
            clusters[i] = cluster_num
        # Store old centroid values
        C_old_store = deepcopy(centroids_curr)
        data.temp_clust = clusters
        
        sse_iter = 0
        for i in range(k):
            points_in_clust = data[data.temp_clust == i][['Latitude', 'Longitude']]
            point_mean = np.mean(points_in_clust, axis=0)
            centroids_curr[i] = point_mean
            lat = point_mean[0]
            lon = point_mean[1]
            matrix = np.array(points_in_clust)
            lats = matrix[:,0]
            lons = matrix[:,1]
            
            for j in range(len(matrix)):
                sse_iter = sse_iter + hav_dist(lat, lon, lats[j], lons[j])
        sse.append(sse_iter)
            
        error = []
        for i in range(k):
            error.append(hav_dist(C_old_store[i][0], C_old_store[i][1], centroids_curr[i][0], centroids_curr[i][1]))
        er = np.linalg.norm(error)
        error_mags.append(er)
        
        
        if verbose:
            print("Magnitude of error: " + str(er))
            print("Iteration took: " + str(time.time()-s2))
            print
    data['CLUSTER_LABEL'] = clusters
    print("Done. Total Time: " + str(time.time() - s1))
    
    return data[['Latitude', 'Longitude', 'CLUSTER_LABEL']], error_mags, sse

def geo_k_means(data, k, alpha = 0.5, cap=1000, verbose=False):
    # pass in data columns that you want to be used for analysis
    # store error for iteration analysis
    error_mags = []
    sse = []
    s1 = time.time()
    
    # initialize centroids
    rand_inds = random.sample(range(0, len(data)), k)
    arr = data.iloc[rand_inds]
    centroids_curr = np.array(arr, dtype=np.float32)
    
    # centroid storage
    centroids_old = np.zeros((k,2))
    # cluster label
    clusters = np.zeros(len(data))
    ideal_error = list(np.zeros(k))
    error = []
    for i in range(k):
        # lats and lons for haversine distance
        error.append(hav_dist(centroids_old[i][0], centroids_old[i][1], centroids_curr[i][0], centroids_curr[i][1]))
    itera = 0
    while not error == ideal_error:
        s2 = time.time()
        itera += 1
        
        # stop early
        if itera > cap:
            break
        
        if verbose:
            print("Iteration: " + str(itera))
        for i in range(len(data)):
            
            distances = []
            lat = data.at[i, 'Latitude']
            lon = data.at[i, 'Longitude']
            all_c = data.at[i, 'All Crime']
            batt = data.at[i, 'Battery']
            theft = data.at[i, 'Theft']
            narc = data.at[i, 'Narcotics']
            for j in range(len(centroids_curr)):
                hav = hav_dist(centroids_curr[j][0], centroids_curr[j][1], lat, lon)
                curr = np.array([all_c, batt, theft, narc])
                cent = np.array([centroids_curr[j][2],centroids_curr[j][3],centroids_curr[j][4],centroids_curr[j][5]])
                vec_dis = np.linalg.norm(curr-cent)
                distances.append((1.0-alpha) * hav + (alpha) * vec_dis)
            cluster_num = np.argmin(distances)
            clusters[i] = cluster_num
        # Store old centroid values
        C_old_store = deepcopy(centroids_curr)
        count_duds = 0
        data.temp_clust = clusters

        sse_iter = 0
        for i in range(k):
            points_in_clust = data[data.temp_clust == i][['Latitude','Longitude','All Crime','Battery','Theft','Narcotics']]
            if len(points_in_clust) > 0:
                centroids_curr[i] = np.mean(points_in_clust, axis=0)
            else:
                count_duds += 1
                rand_ind = random.sample(range(0, len(data)), 1)
                arr = data.iloc[rand_ind]
                centroids_curr[i] = np.array(arr, dtype=np.float32)
            mean = centroids_curr[i]
            lat = mean[0]
            lon = mean[1]
            mean_feat = mean[2:]
            
            matrix = np.array(points_in_clust)
            lats = matrix[:,0]
            lons = matrix[:,1]
            feats = matrix[:,2:]
            
            for j in range(len(matrix)):
                sse_iter = sse_iter + (1.0-alpha)*hav_dist(lat, lon, lats[j], lons[j]) + alpha*np.linalg.norm(mean_feat-feats[j])
        sse.append(sse_iter)
        error = []
        for i in range(k):
            hav = hav_dist(C_old_store[i][0], C_old_store[i][1], centroids_curr[i][0], centroids_curr[i][1])
            v1 = np.array([C_old_store[i][2], C_old_store[i][3], C_old_store[i][4], C_old_store[i][5]])
            v2 = np.array([centroids_curr[i][2], centroids_curr[i][3], centroids_curr[i][4], centroids_curr[i][5]])
            vec = np.linalg.norm(v1-v2)
            error.append(10.0 * hav + 1.0 * vec)
        er = np.linalg.norm(error)
        if verbose:
            print("Magnitude of error: " + str(er))
            print("Iteration took: " + str(time.time()-s2))
            print("Number of issues: " + str(count_duds))
            print
        error_mags.append(er)
    print("Done, Successful Convergence. Total Time: " + str(time.time() - s1))
    data['CLUSTER_LABEL'] = clusters
    return data, error_mags, sse

def geo_k_means_pca_3(data, k, alpha = 0.5, cap=1000, verbose=False):
    # pass in data columns that you want to be used for analysis
    # store error for iteration analysis
    error_mags = []
    sse = []
    s1 = time.time()
    
    # initialize centroids
    rand_inds = random.sample(range(0, len(data)), k)
    arr = data.iloc[rand_inds]
    centroids_curr = np.array(arr, dtype=np.float32)
    
    # centroid storage
    centroids_old = np.zeros((k,2))
    # cluster label
    clusters = np.zeros(len(data))
    ideal_error = list(np.zeros(k))
    error = []
    for i in range(k):
        # lats and lons for haversine distance
        error.append(hav_dist(centroids_old[i][0], centroids_old[i][1], centroids_curr[i][0], centroids_curr[i][1]))
    itera = 0
    while not error == ideal_error:
        s2 = time.time()
        itera += 1
        
        # stop early
        if itera > cap:
            break
        
        if verbose:
            print("Iteration: " + str(itera))
        for i in range(len(data)):
            
            distances = []
            lat = data.at[i, 'Latitude']
            lon = data.at[i, 'Longitude']
            pca_1 = data.at[i, 'PCA_1']
            pca_2 = data.at[i, 'PCA_2']
            pca_3 = data.at[i, 'PCA_3']
            for j in range(len(centroids_curr)):
                hav = hav_dist(centroids_curr[j][0], centroids_curr[j][1], lat, lon)
                curr = np.array([pca_1, pca_2, pca_3])
                cent = np.array([centroids_curr[j][2],centroids_curr[j][3],centroids_curr[j][4]])
                vec_dis = np.linalg.norm(curr-cent)
                distances.append((1.0-alpha) * hav + (alpha) * vec_dis)
            cluster_num = np.argmin(distances)
            clusters[i] = cluster_num
        # Store old centroid values
        C_old_store = deepcopy(centroids_curr)
        count_duds = 0
        data.temp_clust = clusters

        sse_iter = 0
        for i in range(k):
            points_in_clust = data[data.temp_clust == i][['Latitude','Longitude','PCA_1','PCA_2','PCA_3']]
            if len(points_in_clust) > 0:
                centroids_curr[i] = np.mean(points_in_clust, axis=0)
            else:
                count_duds += 1
                rand_ind = random.sample(range(0, len(data)), 1)
                arr = data.iloc[rand_ind]
                centroids_curr[i] = np.array(arr, dtype=np.float32)
            mean = centroids_curr[i]
            lat = mean[0]
            lon = mean[1]
            mean_feat = mean[2:]
            
            matrix = np.array(points_in_clust)
            lats = matrix[:,0]
            lons = matrix[:,1]
            feats = matrix[:,2:]
            
            for j in range(len(matrix)):
                sse_iter = sse_iter + (1.0-alpha)*hav_dist(lat, lon, lats[j], lons[j]) + alpha*np.linalg.norm(mean_feat-feats[j])
        sse.append(sse_iter)
        error = []
        for i in range(k):
            hav = hav_dist(C_old_store[i][0], C_old_store[i][1], centroids_curr[i][0], centroids_curr[i][1])
            v1 = np.array([C_old_store[i][2], C_old_store[i][3], C_old_store[i][4]])
            v2 = np.array([centroids_curr[i][2], centroids_curr[i][3], centroids_curr[i][4]])
            vec = np.linalg.norm(v1-v2)
            error.append(10.0 * hav + 1.0 * vec)
        er = np.linalg.norm(error)
        if verbose:
            print("Magnitude of error: " + str(er))
            print("Iteration took: " + str(time.time()-s2))
            print("Number of issues: " + str(count_duds))
            print
        error_mags.append(er)
    print("Done, Successful Convergence. Total Time: " + str(time.time() - s1))
    data['CLUSTER_LABEL'] = clusters
    return data, error_mags, sse

def geo_k_means_pca_2(data, k, alpha = 0.5, cap=1000, verbose=False):
    # pass in data columns that you want to be used for analysis
    # store error for iteration analysis
    error_mags = []
    sse = []
    s1 = time.time()
    
    # initialize centroids
    rand_inds = random.sample(range(0, len(data)), k)
    arr = data.iloc[rand_inds]
    centroids_curr = np.array(arr, dtype=np.float32)
    
    # centroid storage
    centroids_old = np.zeros((k,2))
    # cluster label
    clusters = np.zeros(len(data))
    ideal_error = list(np.zeros(k))
    error = []
    for i in range(k):
        # lats and lons for haversine distance
        error.append(hav_dist(centroids_old[i][0], centroids_old[i][1], centroids_curr[i][0], centroids_curr[i][1]))
    itera = 0
    while not error == ideal_error:
        s2 = time.time()
        itera += 1
        
        # stop early
        if itera > cap:
            break
        
        if verbose:
            print("Iteration: " + str(itera))
        for i in range(len(data)):
            
            distances = []
            lat = data.at[i, 'Latitude']
            lon = data.at[i, 'Longitude']
            pca_1 = data.at[i, 'PCA_1']
            pca_2 = data.at[i, 'PCA_2']
            for j in range(len(centroids_curr)):
                hav = hav_dist(centroids_curr[j][0], centroids_curr[j][1], lat, lon)
                curr = np.array([pca_1, pca_2])
                cent = np.array([centroids_curr[j][2],centroids_curr[j][3]])
                vec_dis = np.linalg.norm(curr-cent)
                distances.append((1.0-alpha) * hav + (alpha) * vec_dis)
            cluster_num = np.argmin(distances)
            clusters[i] = cluster_num
        # Store old centroid values
        C_old_store = deepcopy(centroids_curr)
        count_duds = 0
        data.temp_clust = clusters

        sse_iter = 0
        for i in range(k):
            points_in_clust = data[data.temp_clust == i][['Latitude','Longitude','PCA_1','PCA_2']]
            if len(points_in_clust) > 0:
                centroids_curr[i] = np.mean(points_in_clust, axis=0)
            else:
                count_duds += 1
                rand_ind = random.sample(range(0, len(data)), 1)
                arr = data.iloc[rand_ind]
                centroids_curr[i] = np.array(arr, dtype=np.float32)
            mean = centroids_curr[i]
            lat = mean[0]
            lon = mean[1]
            mean_feat = mean[2:]
            
            matrix = np.array(points_in_clust)
            lats = matrix[:,0]
            lons = matrix[:,1]
            feats = matrix[:,2:]
            
            for j in range(len(matrix)):
                sse_iter = sse_iter + (1.0-alpha)*hav_dist(lat, lon, lats[j], lons[j]) + alpha*np.linalg.norm(mean_feat-feats[j])
        sse.append(sse_iter)
        error = []
        for i in range(k):
            hav = hav_dist(C_old_store[i][0], C_old_store[i][1], centroids_curr[i][0], centroids_curr[i][1])
            v1 = np.array([C_old_store[i][2], C_old_store[i][3]])
            v2 = np.array([centroids_curr[i][2], centroids_curr[i][3]])
            vec = np.linalg.norm(v1-v2)
            error.append(10.0 * hav + 1.0 * vec)
        er = np.linalg.norm(error)
        if verbose:
            print("Magnitude of error: " + str(er))
            print("Iteration took: " + str(time.time()-s2))
            print("Number of issues: " + str(count_duds))
            print
        error_mags.append(er)
    print("Done, Successful Convergence. Total Time: " + str(time.time() - s1))
    data['CLUSTER_LABEL'] = clusters
    return data, error_mags, sse

def geo_k_means_pca_1(data, k, alpha = 0.5, cap=1000, verbose=False):
    # pass in data columns that you want to be used for analysis
    # store error for iteration analysis
    error_mags = []
    sse = []
    s1 = time.time()
    
    # initialize centroids
    rand_inds = random.sample(range(0, len(data)), k)
    arr = data.iloc[rand_inds]
    centroids_curr = np.array(arr, dtype=np.float32)
    
    # centroid storage
    centroids_old = np.zeros((k,2))
    # cluster label
    clusters = np.zeros(len(data))
    ideal_error = list(np.zeros(k))
    error = []
    for i in range(k):
        # lats and lons for haversine distance
        error.append(hav_dist(centroids_old[i][0], centroids_old[i][1], centroids_curr[i][0], centroids_curr[i][1]))
    itera = 0
    while not error == ideal_error:
        s2 = time.time()
        itera += 1
        
        # stop early
        if itera > cap:
            break
        
        if verbose:
            print("Iteration: " + str(itera))
        for i in range(len(data)):
            
            distances = []
            lat = data.at[i, 'Latitude']
            lon = data.at[i, 'Longitude']
            pca_1 = data.at[i, 'PCA_1']
            for j in range(len(centroids_curr)):
                hav = hav_dist(centroids_curr[j][0], centroids_curr[j][1], lat, lon)
                curr = np.array([pca_1])
                cent = np.array([centroids_curr[j][2]])
                vec_dis = np.linalg.norm(curr-cent)
                distances.append((1.0-alpha) * hav + (alpha) * vec_dis)
            cluster_num = np.argmin(distances)
            clusters[i] = cluster_num
        # Store old centroid values
        C_old_store = deepcopy(centroids_curr)
        count_duds = 0
        data.temp_clust = clusters

        sse_iter = 0
        for i in range(k):
            points_in_clust = data[data.temp_clust == i][['Latitude','Longitude','PCA_1']]
            if len(points_in_clust) > 0:
                centroids_curr[i] = np.mean(points_in_clust, axis=0)
            else:
                count_duds += 1
                rand_ind = random.sample(range(0, len(data)), 1)
                arr = data.iloc[rand_ind]
                centroids_curr[i] = np.array(arr, dtype=np.float32)
            mean = centroids_curr[i]
            lat = mean[0]
            lon = mean[1]
            mean_feat = mean[2:]
            
            matrix = np.array(points_in_clust)
            lats = matrix[:,0]
            lons = matrix[:,1]
            feats = matrix[:,2:]
            
            for j in range(len(matrix)):
                sse_iter = sse_iter + (1.0-alpha)*hav_dist(lat, lon, lats[j], lons[j]) + alpha*np.linalg.norm(mean_feat-feats[j])
        sse.append(sse_iter)
        error = []
        for i in range(k):
            hav = hav_dist(C_old_store[i][0], C_old_store[i][1], centroids_curr[i][0], centroids_curr[i][1])
            v1 = np.array([C_old_store[i][2]])
            v2 = np.array([centroids_curr[i][2]])
            vec = np.linalg.norm(v1-v2)
            error.append(10.0 * hav + 1.0 * vec)
        er = np.linalg.norm(error)
        if verbose:
            print("Magnitude of error: " + str(er))
            print("Iteration took: " + str(time.time()-s2))
            print("Number of issues: " + str(count_duds))
            print
        error_mags.append(er)
    print("Done, Successful Convergence. Total Time: " + str(time.time() - s1))
    data['CLUSTER_LABEL'] = clusters
    return data, error_mags, sse

def regularize(data, cluster_col):
    
    s = time.time()
    lats = set(list(data.Latitude))
    lons = set(list(data.Longitude))
    
    #total_existing = 0
    total_reg_penalty = 0
    #failures = 0
    
    for index, row in data.iterrows():
        
        lat = row.Latitude
        lon = row.Longitude
        nei_identity = row[cluster_col]
        
        s1 = time.time()
        
        try:
            lat_above = np.min([i for i in lats if i > lat])
        except:
            #failures = failures + 1
            lat_above = 0.0
        try:
            lat_below = np.max([i for i in lats if i < lat])
        except:
            #failures = failures + 1
            lat_below = 0.0
        try:
            lon_above = np.min([i for i in lons if i > lon])
        except:
            #failures = failures + 1
            lon_above = 0.0
        try:
            lon_below = np.max([i for i in lons if i < lon])
        except:
            #failures = failures + 1
            lon_below = 0.0
            
        s2 = time.time()
        
        lat_list = [i for i in [lat, lat_above, lat_below] if i != 0.0]
        lon_list = [i for i in [lon, lon_above, lon_below] if i != 0.0]
        
        for latitude in lat_list:
            lat_cut = data[data.Latitude == latitude]
            for longitude in lon_list:
                #allegiance = data.loc[(data.Latitude == latitude) & (data.Longitude == longitude)][cluster_col]
                lon_cut = lat_cut[lat_cut.Longitude == longitude]
                allegiance = list(lon_cut[cluster_col])
                
                if len(allegiance) == 1:
                    #total_existing = total_existing + 1
                    if not allegiance[0] == nei_identity:
                        total_reg_penalty = total_reg_penalty + 1
        
        s3 = time.time()
        
    print("Total time: " + str(time.time() - s))
    print("Regularization Penalty: " + str(total_reg_penalty))
        
    return total_reg_penalty

def silhouette(data, cluster_col, n_clusters):
    cols = ['Latitude', 'Longitude', 'All Crime', 'Battery', 'Assault', 'Theft']
    X = data[cols]
    cluster_labels = data[cluster_col]
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters=" + str(n_clusters) + " The average silhouette_score is: "+ str(silhouette_avg))
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    fig, ax1 = plt.subplots(1, 1)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()
    
    return silhouette_avg

def optimal_k_run(alpha):
    k_list = [80, 85, 90, 95, 100, 105, 110, 115, 120]

    data_for_norm3 = deepcopy(clustering_data)[['Latitude', 'Longitude', 'All Crime', 'Battery', 'Theft', 'Narcotics']]
    for col in ['All Crime', 'Battery', 'Theft', 'Narcotics']:
        data_for_norm3[col] = (data_for_norm3[col]-data_for_norm3[col].mean())/data_for_norm3[col].std()
    #clustering_data = clustering_grid[['Latitude', 'Longitude', 'All Crime', 'Battery', 'Theft', 'Narcotics']]

    sd_list_3 = []
    emd_list_3 = []
    reg_pen_list_3 = []
    final_sse = []
    silhouettes = []

    for k_val in k_list:
    
        df, error, sse = geo_k_means(data_for_norm3[['Latitude', 'Longitude', 'All Crime', 'Battery', 'Theft', 'Narcotics']], k=k_val, alpha=alpha, cap=1000,verbose=False)
        #df, error = pure_geo_k_means_capped(clustering_data[['Latitude', 'Longitude', 'All Crime', 'Battery', 'Theft', 'Narcotics']], k_val, 15)
        str_lab = "Cluster_Label_K_" + str(k_val)
        clustering_grid[str_lab] = [int(i) for i in list(df.CLUSTER_LABEL)]
    
        sns.lmplot(x="Longitude", y="Latitude", data=clustering_grid, fit_reg=False, hue=str_lab, legend=False, scatter_kws={"s": 80}, size=10)
        plt.show()
    
        reg_pen3 = regularize(clustering_grid, str_lab)
        reg_pen_list_3.append(reg_pen3)
        final_sse.append(sse[len(sse)-1])
    
        all_c_emd = emd(clustering_grid, str_lab)
        emd_list_3.append(np.median(all_c_emd))
        all_c_sd = sd(clustering_grid, str_lab)
        sd_list_3.append(np.median(all_c_sd))
        sil_val = silhouette(clustering_grid, str_lab, k_val)
        silhouettes.append(sil_val)
        print("K Value: " + str(k_val) + ", median EMD: " + str(np.median(all_c_emd)))
        print
    
    fig, axs = plt.subplots(1,2, figsize=(16,8))
    axs[0].plot(k_list, final_sse)
    #Regularization = plt.plot(k_list, [x for x in reg_pen_list_3])
    axs[0].set_xlabel("K Value")
    axs[0].set_ylabel("Average All SSE")
    axs[0].set_title("K-Means Clustering Elbow Plot")
    axs[1].plot(k_list, silhouettes)
    axs[1].set_xlabel("K Value")
    axs[1].set_ylabel("Average Silhouette")
    axs[1].set_title("Silhouette Method for Optimal K")
    plt.show()

# EVALUATE MODEL
def evaluate_model(data, clustering_label, save_label, error_mags, sse):
	sns.lmplot(x="Longitude", y="Latitude", data=data, fit_reg=False, hue=clustering_label, legend=False, scatter_kws={"s": 80}, size=10)
	plt.show()

	clustering_grid[save_label] = [int(i) for i in list(data[clustering_label])]
	all_c_emd = emd(clustering_grid, save_label)
	all_c_sd = sd(clustering_grid, save_label)
	print("The average Simple Cluster all crime EMD is: " + str(np.mean(all_c_emd)) + ", the median is: " + str(np.median(all_c_emd)) + ", the maximum is: " + str(np.max(all_c_emd)) + ", and the minimum is: " + str(np.min(all_c_emd)))
	print
	print('-----------------------------------------------------')
	print
	print("The average Simple Cluster all crime Standard Deviation is: " + str(np.mean(all_c_sd)) + ", the median is: " + str(np.median(all_c_sd)) + ", the maximum is: " + str(np.max(all_c_sd)) + ", and the minimum is: " + str(np.min(all_c_sd)))
	print
	fig, axs = plt.subplots(1,2, figsize=(12,6))
	axs[0].plot(range(len(error_mags)), error_mags)
	axs[0].set_xlabel('Iteration Number')
	axs[0].set_ylabel('Magnitude of Iteration Error Vector')
	axs[0].set_title('Convergence Analysis (Using Movement of Centroids)')
	axs[1].plot(range(len(sse)), sse)
	axs[1].set_xlabel('Iteration Number')
	axs[1].set_ylabel('Magnitude of Iteration SSE')
	axs[1].set_title('Convergence Analysis (SSE, Should Decrease)')
	plt.show()

def geo_k_means_97_pca_2(c_g, data, k, alpha=0.5, verbose=False):

    cap=1000

    k = 97
    # pass in data columns that you want to be used for analysis
    # store error for iteration analysis
    error_mags = []
    sse = []
    s1 = time.time()
    
    # initialize centroids
    lats = []
    lons = []
    max_neigh = np.max(list(c_g.Neighborhoods))
    for neigh in range(max_neigh):
        data_neigh = c_g[c_g.Neighborhoods == neigh]
        lats.append(np.mean(list(data_neigh.Latitude)))
        lons.append(np.mean(list(data_neigh.Longitude)))
        
    # initialize centroids based on these points
    centroids_indices = []
    for i in range(len(lats)):
        lat = lats[i]
        lon = lons[i]
        distances = []
        for index, row in c_g.iterrows():
            point_lat = row.Latitude
            point_lon = row.Longitude
            distances.append(hav_dist(lat, lon, point_lat, point_lon))
        min_ind = np.argmin(distances)
        centroids_indices.append(min_ind)
    
    arr = data.iloc[centroids_indices][['Latitude', 'Longitude', 'PCA_1', 'PCA_2']]
    centroids_curr = np.array(arr, dtype=np.float32)
    
    # centroid storage
    centroids_old = np.zeros((k,2))
    # cluster label
    clusters = np.zeros(len(data))
    ideal_error = list(np.zeros(k))
    error = []
    for i in range(k):
        # lats and lons for haversine distance
        error.append(hav_dist(centroids_old[i][0], centroids_old[i][1], centroids_curr[i][0], centroids_curr[i][1]))
    itera = 0
    while not error == ideal_error:
        s2 = time.time()
        itera += 1
        
        # stop early
        if itera > cap:
            break
        
        if verbose:
            print("Iteration: " + str(itera))
        for i in range(len(data)):
            
            distances = []
            lat = data.at[i, 'Latitude']
            lon = data.at[i, 'Longitude']
            pca_1 = data.at[i, 'PCA_1']
            pca_2 = data.at[i, 'PCA_2']
            for j in range(len(centroids_curr)):
                hav = hav_dist(centroids_curr[j][0], centroids_curr[j][1], lat, lon)
                curr = np.array([pca_1, pca_2])
                cent = np.array([centroids_curr[j][2],centroids_curr[j][3]])
                vec_dis = np.linalg.norm(curr-cent)
                distances.append((1.0-alpha) * hav + (alpha) * vec_dis)
            cluster_num = np.argmin(distances)
            clusters[i] = cluster_num
        # Store old centroid values
        C_old_store = deepcopy(centroids_curr)
        count_duds = 0
        data.temp_clust = clusters

        sse_iter = 0
        for i in range(k):
            points_in_clust = data[data.temp_clust == i][['Latitude','Longitude','PCA_1','PCA_2']]
            if len(points_in_clust) > 0:
                centroids_curr[i] = np.mean(points_in_clust, axis=0)
            else:
                count_duds += 1
                rand_ind = random.sample(range(0, len(data)), 1)
                arr = data.iloc[rand_ind]
                centroids_curr[i] = np.array(arr, dtype=np.float32)
            mean = centroids_curr[i]
            lat = mean[0]
            lon = mean[1]
            mean_feat = mean[2:]
            
            matrix = np.array(points_in_clust)
            lats = matrix[:,0]
            lons = matrix[:,1]
            feats = matrix[:,2:]
            
            for j in range(len(matrix)):
                sse_iter = sse_iter + (1.0-alpha)*hav_dist(lat, lon, lats[j], lons[j]) + alpha*np.linalg.norm(mean_feat-feats[j])
        sse.append(sse_iter)
        error = []
        for i in range(k):
            hav = hav_dist(C_old_store[i][0], C_old_store[i][1], centroids_curr[i][0], centroids_curr[i][1])
            v1 = np.array([C_old_store[i][2], C_old_store[i][3]])
            v2 = np.array([centroids_curr[i][2], centroids_curr[i][3]])
            vec = np.linalg.norm(v1-v2)
            error.append(10.0 * hav + 1.0 * vec)
        er = np.linalg.norm(error)
        if verbose:
            print("Magnitude of error: " + str(er))
            print("Iteration took: " + str(time.time()-s2))
            print("Number of issues: " + str(count_duds))
            print
        error_mags.append(er)
    print("Done, Successful Convergence. Total Time: " + str(time.time() - s1))
    data['CLUSTER_LABEL'] = clusters
    return data, error_mags, sse

def geo_k_means_95_pca_2(c_g, data, alpha=0.5, k=96, verbose=False):

    cap=1000

    # pass in data columns that you want to be used for analysis
    # store error for iteration analysis
    error_mags = []
    sse = []
    s1 = time.time()
    
    # initialize centroids
    X = c_g[["lat", "lon"]]
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    y_km = kmeans.fit_predict(X)
    c_g['Temp_Cluster'] = y_km

    lats = []
    lons = []

    max_neigh = np.max(list(c_g.Temp_Cluster))
    for neigh in range(max_neigh+1):
        data_neigh = c_g[c_g.Temp_Cluster == neigh]
        if len(data_neigh) > 0:
            lats.append(np.mean(list(data_neigh.Latitude)))
            lons.append(np.mean(list(data_neigh.Longitude)))
    k = len(lats)
    
    # initialize centroids based on these points
    centroids_indices = []
    for i in range(len(lats)):
        lat = lats[i]
        lon = lons[i]
        distances = []
        for index, row in c_g.iterrows():
            point_lat = row.Latitude
            point_lon = row.Longitude
            distances.append(hav_dist(lat, lon, point_lat, point_lon))
        min_ind = np.argmin(distances)
        centroids_indices.append(min_ind)
    
    arr = data.iloc[centroids_indices][['Latitude', 'Longitude', 'PCA_1', 'PCA_2']]
    centroids_curr = np.array(arr, dtype=np.float32)
    #print(centroids_curr)
    
    #lats = list(arr.Latitude)
    #lons = list(arr.Longitude)

    #plt.scatter(lons, lats, s=10)
    #plt.title("Centroids Corresponding to Center of Chicago's 97 Neighborhoods")
    #plt.show()
    
    # centroid storage
    centroids_old = np.zeros((k,2))
    # cluster label
    clusters = np.zeros(len(data))
    ideal_error = list(np.zeros(k))
    error = []
    for i in range(k):
        # lats and lons for haversine distance
        error.append(hav_dist(centroids_old[i][0], centroids_old[i][1], centroids_curr[i][0], centroids_curr[i][1]))
    itera = 0
    while not error == ideal_error:
        s2 = time.time()
        itera += 1
        
        # stop early
        if itera > cap:
            break
        
        if verbose:
            print("Iteration: " + str(itera))
        for i in range(len(data)):
            
            distances = []
            lat = data.at[i, 'Latitude']
            lon = data.at[i, 'Longitude']
            pca_1 = data.at[i, 'PCA_1']
            pca_2 = data.at[i, 'PCA_2']
            for j in range(len(centroids_curr)):
                hav = hav_dist(centroids_curr[j][0], centroids_curr[j][1], lat, lon)
                curr = np.array([pca_1, pca_2])
                cent = np.array([centroids_curr[j][2],centroids_curr[j][3]])
                vec_dis = np.linalg.norm(curr-cent)
                distances.append((1.0-alpha) * hav + (alpha) * vec_dis)
            cluster_num = np.argmin(distances)
            clusters[i] = cluster_num
        # Store old centroid values
        C_old_store = deepcopy(centroids_curr)
        count_duds = 0
        data.temp_clust = clusters

        sse_iter = 0
        for i in range(k):
            points_in_clust = data[data.temp_clust == i][['Latitude','Longitude','PCA_1','PCA_2']]
            if len(points_in_clust) > 0:
                centroids_curr[i] = np.mean(points_in_clust, axis=0)
            else:
                count_duds += 1
                rand_ind = random.sample(range(0, len(data)), 1)
                arr = data.iloc[rand_ind]
                centroids_curr[i] = np.array(arr, dtype=np.float32)
                print("")
            mean = centroids_curr[i]
            lat = mean[0]
            lon = mean[1]
            mean_feat = mean[2:]
            
            matrix = np.array(points_in_clust)
            lats = matrix[:,0]
            lons = matrix[:,1]
            feats = matrix[:,2:]
            
            for j in range(len(matrix)):
                sse_iter = sse_iter + (1.0-alpha)*hav_dist(lat, lon, lats[j], lons[j]) + alpha*np.linalg.norm(mean_feat-feats[j])
        sse.append(sse_iter)
        error = []
        for i in range(k):
            hav = hav_dist(C_old_store[i][0], C_old_store[i][1], centroids_curr[i][0], centroids_curr[i][1])
            v1 = np.array([C_old_store[i][2], C_old_store[i][3]])
            v2 = np.array([centroids_curr[i][2], centroids_curr[i][3]])
            vec = np.linalg.norm(v1-v2)
            error.append(10.0 * hav + 1.0 * vec)
        er = np.linalg.norm(error)
        if verbose:
            print("Magnitude of error: " + str(er))
            print("Iteration took: " + str(time.time()-s2))
            print("Number of issues: " + str(count_duds))
            print
        error_mags.append(er)
    print("Done, Successful Convergence. Total Time: " + str(time.time() - s1))
    data['CLUSTER_LABEL'] = clusters
    return data, error_mags, sse

