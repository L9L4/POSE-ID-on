import numpy as np
import copy
from copy import deepcopy
import math

class MatchingClass2():

  links = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[1,8],[7,6],[8,9],
           [8,12],[9,10],[10,11],[12,13],[13,14]]
  new_order = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10,  11]
  weight = 1

  def __init__(self, pose_a, pose_b):
    self.pose_a = pose_a
    self.pose_b = pose_b

  def ang(self, v):
      if math.atan2(v[1], v[0]) > 0:
          beta = math.atan2(v[1], v[0])
      else: 
          beta = math.atan2(v[1], v[0]) + 2*np.pi
      return beta

  def diff_angs(self, ang1, ang2):
      if ang1 - ang2 < 0:
          diff = ang1 - ang2 + 2*np.pi
      else:
          diff = ang1 - ang2
      
      return diff        

  def angolo(self, joint_a, joint_b, joint_c, joint_d):
    v1 = np.array(joint_a) - np.array(joint_b)
    v2 = np.array(joint_c) - np.array(joint_d)
    return self.diff_angs(self.ang(v2),self.ang(v1))

  def pose_mirroring(self, pose):
    new_pose = [pose[i] for i in self.new_order]
    M = max([new_pose[n][0] for n in range(len(new_pose))])

    mirrored_pose = copy.deepcopy(new_pose)

    for i in range(len(mirrored_pose)):
      mirrored_pose[i][0] = - new_pose[i][0] + M

    return mirrored_pose

  def turn_pose(self, pose):
    M = max([pose[n][0] for n in range(len(pose))])

    t_pose = copy.deepcopy(pose)

    for i in range(len(t_pose)):
      t_pose[i][0] = - pose[i][0] + M

    return t_pose

  def calcolo_angoli(self, joints):
    lista_angoli = []
    for i in range(len(self.links)):
      for j in range(i+1,len(self.links)):
        inter = [value for value in self.links[i] if value in self.links[j]]
        ang = self.angolo(joints[self.links[i][0]], joints[self.links[i][1]],
                          joints[self.links[j][0]], joints[self.links[j][1]])
        if len(inter) == 0:
          lista_angoli.append(ang)
        else:
          lista_angoli.append(ang*self.weight)

    return np.array(lista_angoli)

  def diff(self, l1, l2):
    diffs =  []
    for i in range(len(l1)):
      distance = 1-np.cos(l1-l2)
      diffs.append(distance)
    return np.mean(diffs)

  def loss(self):
    ang_pose_a = self.calcolo_angoli(self.pose_a)
    pose_mirror_a = self.pose_mirroring(self.pose_a)
    ang_pose_mirr_a = self.calcolo_angoli(pose_mirror_a)
    pose_t_a = self.turn_pose(self.pose_a)
    ang_pose_t_a = self.calcolo_angoli(pose_t_a)
    ang_pose_b = self.calcolo_angoli(self.pose_b)
    return min(self.diff(np.array(ang_pose_a), np.array(ang_pose_b)), 
               self.diff(np.array(ang_pose_mirr_a), np.array(ang_pose_b)), 
               self.diff(np.array(ang_pose_t_a), np.array(ang_pose_b)))