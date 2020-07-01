import math
import numpy as np

def ang(v):
    if math.atan2(v[1], v[0]) > 0:
        beta = math.atan2(v[1], v[0])
    else: 
        beta = math.atan2(v[1], v[0]) + 2*np.pi
    return beta

def diff_angs(ang1, ang2):
    if ang1 - ang2 < 0:
        diff = ang1 - ang2 + 2*np.pi
    else:
        diff = ang1 - ang2
    
    return diff        

def diff(l1, l2):
  diff_1 = np.abs(l1 - l2)
  if diff_1 <= np.pi:
    diff_2 = diff_1
  else:
    diff_2 = np.min([np.abs(l1 - l2 - 2*np.pi), np.abs(l1 - l2 + 2*np.pi)])   
  return diff_2

def angolo(joint_a, joint_b, joint_c, joint_d):
    v1 = np.array(joint_a) - np.array(joint_b)
    v2 = np.array(joint_c) - np.array(joint_d)
    return diff_angs(ang(v2),ang(v1))

def calcolo_angoli_cons(joints):
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
	punto_medio = [np.mean(np.array(posa)[:,0]), np.mean(np.array(posa)[:,1])]
	dista = 0

	for punto in posa:
	  dista += np.power(punto[0]-punto_medio[0], 2) + np.power(punto[1]-punto_medio[1], 2)
	dista = dista/len(posa)
	return np.sqrt(dista)

def dist(a,b):
    return np.sqrt((b[0]-a[0])**2+(b[1]-a[1])**2)

def calc_dist(X):
  links = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[1,8],[7,6],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14]]
  r = calc_radius(X)
  dist_X = []
  for link in links:
    dista = dist(X[link[0]], X[link[1]])/r
    dist_X.append(dista)
  return dist_X

def dist_medie(dict_joints):
    dist_joints = np.zeros((len(dict_joints),14))

    i = 0
    for key in list(dict_joints.keys()):
        dist_joints[i] = calc_dist(dict_joints[key])
        i += 1

    dist_medie = np.mean(dist_joints,0)
    return 100*dist_medie