import math
import copy
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import os 
import json

def ang(v):
    
    """
    Compute the angle of a vector v with respect to to the x axis in the range [0, 2*pi].
    """
    
    if math.atan2(v[1], v[0]) > 0:
        beta = math.atan2(v[1], v[0])
    else: 
        beta = math.atan2(v[1], v[0]) + 2*np.pi
    return beta

def diff_angs(ang1, ang2):
    
    """
    Compute the difference between two angles ang1, ang2 in the range [0, 2*pi].
    """ 
    
    if ang1 - ang2 < 0:
        diff = ang1 - ang2 + 2*np.pi
    else:
        diff = ang1 - ang2
    
    return diff        

def diff(l1, l2):
  
  """
  Compute a special kind of distance between two angles l1, l2 in the range [0,2].
  """ 
  
  distance = 1-np.cos(l1-l2)
  return distance

def angolo(joint_a, joint_b, joint_c, joint_d):
    
    """
    Compute the angle between two vectors v1, v2 where: 
    v1 = joint_a-joint_b
    v2 = joint_c-joint_d
    """
    
    v1 = np.array(joint_a) - np.array(joint_b)
    v2 = np.array(joint_c) - np.array(joint_d)
    return diff_angs(ang(v2),ang(v1))

def calcolo_angoli_cons(joints):
  
  """
  Compute the angles between the limbs (of a given pose) which share a joint.
  Args:
    joints: the input pose
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

def calc_radius(posa):
  
  """
  Compute the pose radius of gyration. https://en.wikipedia.org/wiki/Radius_of_gyration
  Args:
    posa: the input pose
  """
  
  punto_medio = [np.mean(np.array(posa)[:,0]), np.mean(np.array(posa)[:,1])]
  dista = 0

  for punto in posa:
    dista += np.power(punto[0]-punto_medio[0], 2) + np.power(punto[1]-punto_medio[1], 2)
  dista = dista/len(posa)
  return np.sqrt(dista)

def dist(a,b):
  
  """
  Compute Euclidean distance between a and b.
  """
  
  return np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)

def calc_dist(X):
  
  """
  Compute the lenght of each limb in a given pose X.
  """
  
  links = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[1,8],[7,6],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14]]
  r = calc_radius(X)
  dist_X = []
  for link in links:
    dista = dist(X[link[0]], X[link[1]])/r
    dist_X.append(dista)
  return dist_X

def dist_medie(dict_joints):
  
  """
  Compute the average lenght of each limb in a given dataset, composed by all the poses in dict_joints.
  """
  
  dist_joints = np.zeros((len(dict_joints),14))

  i = 0
  for key in list(dict_joints.keys()):
      dist_joints[i] = calc_dist(dict_joints[key])
      i += 1

  dist_medie = np.mean(dist_joints,0)
  return 100*dist_medie

def pose_mirroring(pose):
  
  """
  Mirror the pose.
  """
  
  new_order = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10,  11]
  new_pose = [pose[i] for i in new_order]
  M = max([new_pose[n][0] for n in range(len(new_pose))])

  mirrored_pose = copy.deepcopy(new_pose)

  for i in range(len(mirrored_pose)):
    mirrored_pose[i][0] = - new_pose[i][0] + M

  return mirrored_pose

def turn_pose(pose):
  
  """
  Turn the pose.
  """
  
  M = max([pose[n][0] for n in range(len(pose))])

  t_pose = copy.deepcopy(pose)

  for i in range(len(t_pose)):
    t_pose[i][0] = - pose[i][0] + M

  return t_pose

def load_poses(dir_im, dir_joints):
  
  """
  Load the poses from a given dir_joints directory, based on the name of the respective images in the dir_im directory.
  The conventions of joint and image names are based on the OpenPose ones. https://github.com/CMU-Perceptual-Computing-Lab/openpose
  """
  
  dict_joints = {}
  dict_joints_SR_destrorso = {}
  for posa in tqdm(os.listdir(dir_im)):
      posa1 = posa[:-13]
      posa = posa1 + "_keypoints.json"
      try:
          file = dir_joints+'/'+posa
          if os.path.isfile(file):
            with open(file) as f:
              data = json.load(f)
            prova = data['people'][0]['pose_keypoints_2d']
            punti = []
            punti1 = []
            i = 0
            max_y = 0.0
            while i < 75:
              x = prova[i]
              y1 = prova[i+1]
              y2 = -prova[i+1]
              punto = [x,y1]
              punto1 = [x,y2]
              punti.append(punto)
              punti1.append(punto1)
              i += 3
              if np.abs(y1)> max_y:
                  max_y = np.abs(y1)
            punti = punti[:15]
            punti1 = punti1[:15]
            for i in range(len(punti1)):
              punti1[i][1] += max_y

            dict_joints[posa1] = punti
            dict_joints_SR_destrorso[posa1] = punti1
      except:
          print(posa)
  return dict_joints, dict_joints_SR_destrorso