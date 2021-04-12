import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from clustering.Clustering import *
from utils.utils import *
from methods.Second_method import *


def create_features(dict_joints_SR_destrorso):
	
	"""
	For each pose in dict_joints_SR_destrorso, compute the 91 features with Matching Class 2 method.
	Output:
		x: array of features
		df: dataframe of features
	"""
	
	features = {}

	for i in dict_joints_SR_destrorso.keys():
	  sample = dict_joints_SR_destrorso[i]
	  MC = MatchingClass2(sample, sample)
	  features[i] = MC.calcolo_angoli(sample)

	coordinates = {}

	for j in range(91):
	  dict_key = "x" + str(j)
	  coordinates[dict_key] = []
	  for i in features.keys():
	    coordinates[dict_key].append(features[i][j])

	df = pd.DataFrame(coordinates,columns=coordinates.keys(), index = dict_joints_SR_destrorso.keys())

	x = df.values

	return x, df

def clustering(x, df, n_clusters = 10, distance = 'angular', method = 'K-medians'):
  
  """
  Do the clustering, based on the 91 features.
  Args:
	  x: array of features
	  df: dataframe of features
	  n_clusters: number of clusters
	  distance: could be 'angular' or 'euclidean';
      method: could be 'K-medians', 'K-means', 'Hierarchical'
  Output:
	  new_df: the labeled dataframe, according to the clustering algorithm
	  relevant_features_cs: a list with the relevant features (angles of the consecutive limbs) of the centroids
	  cs: dictionary with the centroid features 
  """

  relevant_features_id = [0,3,5,13,15,17,25,46,47,56,64,65,76,77,83,85,90]
  keys_dict = ['0-1', '0-4', '0-6', '1-2', '1-4', '1-6', '2-3', '4-5', '4-6', '5-7', '6-8', '6-9', '8-9', '8-10', '9-12', '10-11', '12-13']

  clustering_ = Clustering(k = n_clusters, distance = distance, method = method)
  cs, cls = clustering_.fit(x)

  assert len(list(cls.keys())) == n_clusters
  
  d = pd.DataFrame()
  l = []
  for i in range(n_clusters):
    df1 = pd.DataFrame(cls[i])
    d = pd.concat([d,df1], sort = False)
    l += [i]*len(cls[i])

  d.columns = df.columns
  d.insert(91, 'label', l)

  new_df = df.reset_index().merge(d).set_index('index')

  relevant_features_cs = []
  if method == 'Hierarchical':
  	pass
  else:
  	for i in range(len(cs)):
  		d = {}
  		cs_rf = cs[i][relevant_features_id]
  		for k in range(len(keys_dict)):
  			d[keys_dict[k]] = cs_rf[k]
  		relevant_features_cs.append(d)

  return new_df, relevant_features_cs, cs

def ric_posa(relevant_features_cs, cluster, output_folder, dists, save = True):
  
  """
  Compute and plot the pose reconstructed from a centroid obtained with clustering.
  Args:
  	relevant_features_cs: a list with the relevant features (angles of the consecutive limbs) of the centroids
  	cluster: cluster ID
  	output_folder: output folder
  	dists: the average lenght of each limb, computed on the basis of the whole dataset
  	save: if True, save the reconstructed pose as image
  """

  links = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[1,8],[7,6],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14]]

  dict_an = relevant_features_cs[cluster]
  P0 = np.array([500.0, 500.0])
  X1 = 500.0
  #Scegliamo di avere Y1 sotto ad X1
  Y1 = - np.sqrt(dists[0]**2 - (X1 - P0[0])**2) + P0[1]
  P1 = np.array([X1,Y1])

  rec_pose = np.zeros((15,2))
  rec_pose[0] = P0
  rec_pose[1] = P1

  for i in list(dict_an.keys()):
      link0, link1 = i.split("-")
      link0, link1 = int(link0), int(link1)
      P0 = rec_pose[links[link0][0]]
      P1 = rec_pose[links[link0][1]]
      angoletto = ang(P1 - P0)
      
      alfa = dict_an[i] + angoletto
      
      if alfa > 2*np.pi:
          alfa -= 2*np.pi
      else:
          pass

      if rec_pose[links[link1][0]][0] == 0.0:
          X_t = rec_pose[links[link1][1]][0]
          Y_t = rec_pose[links[link1][1]][1]
          rec_pose[links[link1][0]][0] = -np.cos(alfa)*dists[link1] + X_t
          rec_pose[links[link1][0]][1] = -np.sin(alfa)*dists[link1] + Y_t
      else:
          X_t = rec_pose[links[link1][0]][0]
          Y_t = rec_pose[links[link1][0]][1]   
          rec_pose[links[link1][1]][0] = np.cos(alfa)*dists[link1] + X_t
          rec_pose[links[link1][1]][1] = np.sin(alfa)*dists[link1] + Y_t
  if save == True:
    fig, ax = plt.subplots(figsize = (10,15))

    if not os.path.exists(output_folder):
      os.makedirs(output_folder)

    for n in range(len(rec_pose)):
        plt.plot(rec_pose[n][0], rec_pose[n][1], 'ro')
        ax.annotate(n, (rec_pose[n][0], rec_pose[n][1]))
        ax.set_aspect(aspect = "equal")   
    for l in range(len(links)):
        p1, p2 = links[l]
        plt.plot([rec_pose[p1][0], rec_pose[p2][0]],[rec_pose[p1][1], rec_pose[p2][1]], '-')
    
    plt.savefig(output_folder + os.sep + "{}.png".format(cluster))

    plt.close()
  else:
    pass
  return rec_pose

def clustering_several_n(list_n_clusters, dict_):
	
	"""
	Compute the clustering algorithm for all the values contained in list_n_clusters.
	Args:
		list_n_clusters: list with the number of clusters for each clustering
		dict_: dictionary with all the poses
	Output:
		df_cls: a dictionary composed by the dataframes (containing the statue names, their features and their cluster labels) for each clustering
		all_poses: a dictionary composed by the reconstructed poses of each clustering
	"""
	
	root = os.getcwd() + os.sep + "Cluster"
	dists = dist_medie(dict_)

	if not os.path.exists(root):
	    os.makedirs(root)

	trials = np.array(list_n_clusters)

	all_poses = {}
	df_cl = {}

	for n_c in trials:
	  n_c_poses = []
	  new_df, relevant_features_cs, cs = clustering(create_features(dict_)[0], create_features(dict_)[1], n_c)
	  out_fold = root + os.sep + str(n_c)
	  if not os.path.exists(out_fold):
	    os.makedirs(out_fold)
	  with open(out_fold + os.sep + str(n_c) + ".txt", "w") as f:
	    for i in range(n_c):
	      rec_p = ric_posa(relevant_features_cs, i, out_fold, dists = dists)
	      n_c_poses.append(rec_p)
	      f.write(str(list(new_df[new_df['label'] == i].index)))
	      f.write("\n\n")
	  df_cl[n_c] = new_df
	  all_poses[n_c] = n_c_poses
	  print("Clustering with {} clusters: done".format(n_c))
	return df_cl, all_poses

def mean_reconstruction_error(n_clusters, dict_joints_SR_destrorso):
	
	"""
	For all the centroids obtained from clustering with n_clusters clusters, compute the difference between the centroid feature vector and the one
	obtained from the reconstructed pose.
	Args:
		n_clusters: number of clusters 
		dict_joints_SR_destrorso: pose dictionary
	Outputs:
		MRE: mean reconstruction error with respect to the n_clusters
		mean_errors: list of reconstruction errors, each one specific for a given centroid 
	"""
	
	df_clustering, relevant_features_centroids, centroids = clustering(create_features(dict_joints_SR_destrorso)[0], 
	                                                            create_features(dict_joints_SR_destrorso)[1], 
	                                                            n_clusters)
	dists = dist_medie(dict_joints_SR_destrorso)
	mean_errors = []
	for j in range(len(centroids)):
	    reconstructed_pose = ric_posa(relevant_features_centroids, j, dists = dists, output_folder = "./.", save = False)
	    MC = MatchingClass2(reconstructed_pose, reconstructed_pose)
	    features = MC.calcolo_angoli(reconstructed_pose)
	    
	    diffs = []
	    for i in range(features.shape[0]):
	      diff_1 = np.abs(features[i] - centroids[j][i])
	      if diff_1 <= np.pi:
	        diff_2 = diff_1
	      else:
	        diff_2 = np.min([np.abs(features[i] - centroids[j][i] - 2*np.pi), np.abs(features[i] - centroids[j][i]+ 2*np.pi)])  
	      diffs.append(diff_2)

	    diffs = np.array(diffs)
	    mean_error = (np.sum(diffs)/91)*180/np.pi
	    mean_errors.append(mean_error)
	MRE = np.mean(np.array(mean_errors))
	print("Mean reconstruction error = {}°".format(MRE))
	return MRE, mean_errors