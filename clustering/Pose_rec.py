import numpy as np
import math
import matplotlib.pyplot as plt
import os
from clustering.KMeans import *

def ang(v):
    """
    Computes the angle of a vector v w.r.t. to the x axis in the range [0, 2*pi]
    """
    if math.atan2(v[1], v[0]) > 0:
        beta = math.atan2(v[1], v[0])
    else: 
        beta = math.atan2(v[1], v[0]) + 2*np.pi
    return beta

def diff_angs(ang1, ang2):
    """
    Computes the difference of two angles ang1, ang2 in the range [0, 2*pi]
    """ 
    if ang1 - ang2 < 0:
        diff = ang1 - ang2 + 2*np.pi
    else:
        diff = ang1 - ang2
    
    return diff        

def diff(l1, l2):
    """
    Computes the distance between two angles l1, l2 in the range [0,2]
    """ 
  distance = 1-np.cos(l1-l2)
  return distance

def angolo(joint_a, joint_b, joint_c, joint_d):
    """
    Computes the angle between two vector v1, v2 where: 
    v1 = joint_a-joint_b
    v2 = joint_c-joint_d
    """
    v1 = np.array(joint_a) - np.array(joint_b)
    v2 = np.array(joint_c) - np.array(joint_d)
    return diff_angs(ang(v2),ang(v1))

def calcolo_angoli_cons(joints):
  """
  Computes the angles between limbs sharing a joint for a given pose.
  """
    lista_angoli = []
    dict_an = {}
    for i in range(len(links)):
      for j in range(i+1,len(links)):
        inter = [value for value in links[i] if value in links[j]]
        ang = angolo(joints[links[i][0]], joints[links[i][1]],
                          joints[links[j][0]], joints[links[j][1]])
        if len(inter) == 0:
            pass
        else:
          lista_angoli.append(ang)
          dict_an['{}-{}'.format(i,j)] = ang

            
    return [np.array(lista_angoli), dict_an]

def k_means(x, n_clusters):
  """
  Do the KMeans clustering, based on the 91 features.
  Args:
    x: array of features
    n_clusters: number of clusters
  Output:
    new_df: the labeled dataframe, according to the kmeans alg
    relevant_features_cs: a list with the relevant features (angles of the consecutive limbs) of the centroids
    cs: dictionary with the features of the centroids
  """
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

def ric_posa(relevant_features_cs, cluster, output_folder, links = links, dists = dists):
  """
  Compute and plot the pose of the centroid of kmeans.
  Args:
    relevant_features_cs: a list with the relevant features (angles of the consecutive limbs) of the centroids
    cluster: cluster ID
    output_folder: output folder
    dists: the average lenghts of each limb
    links = list of limbs composed by a pair of joints
  """
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

  fig, ax = plt.subplots(figsize = (5,5))
  
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for n in range(len(rec_pose)):
      plt.plot(rec_pose[n][0], rec_pose[n][1], 'ro')
      ax.annotate(n, (rec_pose[n][0], rec_pose[n][1]))
      ax.set_aspect(aspect = "equal")   
  for l in range(len(links)):
      p1, p2 = links[l]
      plt.plot([rec_pose[p1][0], rec_pose[p2][0]],[rec_pose[p1][1], rec_pose[p2][1]], '-')
      #plt.savefig(output_folder + os.sep + "{}.png".format(cluster))
  return rec_pose