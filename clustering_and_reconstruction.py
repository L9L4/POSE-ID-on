import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
from KMeans import *
from utils import *
from Second_method import *


def create_features(dict_joints_SR_destrorso):

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

def k_means(x, df, n_clusters = 10):

  relevant_features_id = [0,3,5,13,15,17,25,46,47,56,64,65,76,77,83,85,90]
  keys_dict = ['0-1', '0-4', '0-6', '1-2', '1-4', '1-6', '2-3', '4-5', '4-6', '5-7', '6-8', '6-9', '8-9', '8-10', '9-12', '10-11', '12-13']

  kmeans = K_Means(k = n_clusters)
  cs, cls = kmeans.fit(x)
  d = pd.DataFrame()
  l = []
  for i in range(len(cs)):
    df1 = pd.DataFrame(cls[i])
    d = pd.concat([d,df1], sort = False)
    l += [i]*len(cls[i])

  d.columns = df.columns
  d.insert(91, 'label', l)

  new_df = df.reset_index().merge(d).set_index('index')

  assert len(cs) == n_clusters

  relevant_features_cs = []
  for i in range(len(cs)):
    d = {}
    cs_rf = cs[i][relevant_features_id]
    for k in range(len(keys_dict)):
      d[keys_dict[k]] = cs_rf[k]
    relevant_features_cs.append(d)

  return new_df, relevant_features_cs, cs

def ric_posa(relevant_features_cs, cluster, output_folder, dists):

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
  return rec_pose

def clustering_several_n(list_n_clusters, dict_):
    root = os.getcwd() + os.sep + "Cluster"
    dists = dist_medie(dict_)
    
    if not os.path.exists(root):
        os.makedirs(root)
    
    trials = np.array(list_n_clusters)

    for n_c in trials:
      new_df, relevant_features_cs, cs = k_means(create_features(dict_)[0], create_features(dict_)[1], n_c)
      out_fold = root + os.sep + str(n_c)
      if not os.path.exists(out_fold):
        os.makedirs(out_fold)
      with open(out_fold + os.sep + str(n_c) + ".txt", "w") as f:
        for i in range(n_c):
          rec_p = ric_posa(relevant_features_cs, i, out_fold, dists = dists)
          f.write(str(list(new_df[new_df['label'] == i].index)))
          f.write("\n\n")
      print("Clustering with {} clusters: done".format(n_c))
    return