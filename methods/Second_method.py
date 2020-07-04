import numpy as np
import copy
from copy import deepcopy
import math
from tqdm import tqdm
import operator

class MatchingClass2():
    
  """
  Matching Class Method 2.
  """

  links = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[1,8],[7,6],[8,9],
           [8,12],[9,10],[10,11],[12,13],[13,14]]
  new_order = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10,  11]
  weight = 1

  def __init__(self, pose_a, pose_b):
    self.pose_a = pose_a
    self.pose_b = pose_b

  def ang(self, v):
      
      """
      Compute the angle of a vector v with respect to to the x axis in the range [0, 2*pi].
      """

      if math.atan2(v[1], v[0]) > 0:
          beta = math.atan2(v[1], v[0])
      else: 
          beta = math.atan2(v[1], v[0]) + 2*np.pi
      return beta

  def diff_angs(self, ang1, ang2):
      
      """
      Compute the difference between two angles ang1, ang2 in the range [0, 2*pi].
      """ 

      if ang1 - ang2 < 0:
          diff = ang1 - ang2 + 2*np.pi
      else:
          diff = ang1 - ang2
      
      return diff        

  def angolo(self, joint_a, joint_b, joint_c, joint_d):
    
    """
    Compute the angle between two vectors v1, v2 where: 
    v1 = joint_a-joint_b
    v2 = joint_c-joint_d
    """

    v1 = np.array(joint_a) - np.array(joint_b)
    v2 = np.array(joint_c) - np.array(joint_d)
    return self.diff_angs(self.ang(v2),self.ang(v1))

  def pose_mirroring(self, pose):
    
    """
    Mirror the pose.
    """

    new_pose = [pose[i] for i in self.new_order]
    M = max([new_pose[n][0] for n in range(len(new_pose))])

    mirrored_pose = copy.deepcopy(new_pose)

    for i in range(len(mirrored_pose)):
      mirrored_pose[i][0] = - new_pose[i][0] + M

    return mirrored_pose

  def turn_pose(self, pose):
    
    """
    Turn the pose.
    """
    
    M = max([pose[n][0] for n in range(len(pose))])

    t_pose = copy.deepcopy(pose)

    for i in range(len(t_pose)):
      t_pose[i][0] = - pose[i][0] + M

    return t_pose

  def calcolo_angoli(self, joints):
    
    """
    Compute the angles among all the limbs for a given pose.
    Args:
      joints: the input pose
    """
    
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
    
    """
    Compute the distance between two feature vectors l1, l2.
    """ 
    
    diffs =  []
    for i in range(len(l1)):
      distance = 1-np.cos(l1-l2)
      diffs.append(distance)
    return np.mean(diffs)

  def loss(self):
    
    """
    Compute the difference (loss) between the query pose and a compared one. 
    The function returns the minimum difference between the pose b and the pose a (itself, mirrored and turned)
    """

    ang_pose_a = self.calcolo_angoli(self.pose_a)
    pose_mirror_a = self.pose_mirroring(self.pose_a)
    ang_pose_mirr_a = self.calcolo_angoli(pose_mirror_a)
    pose_t_a = self.turn_pose(self.pose_a)
    ang_pose_t_a = self.calcolo_angoli(pose_t_a)
    ang_pose_b = self.calcolo_angoli(self.pose_b)
    return min(self.diff(np.array(ang_pose_a), np.array(ang_pose_b)), 
               self.diff(np.array(ang_pose_mirr_a), np.array(ang_pose_b)), 
               self.diff(np.array(ang_pose_t_a), np.array(ang_pose_b)))

def second_method_app(dict_joints_SR_destrorso):
    
    """
    Apply the second method to search for the most similar poses to a given one (within the dictionary dict_joints_SR_destrorso).
    Output:
        best_worst_cases_2: dictionary with:
            keys: poses of the input dictionary
            values: list with the 5 closest poses (inluding the query) and the farthest one
    """
    
    best_worst_cases_2 = {}
    for i in tqdm(range(len(dict_joints_SR_destrorso))):
        sample = dict_joints_SR_destrorso[list(dict_joints_SR_destrorso.keys())[i]]
        dict_losses = {}
        for key in list(dict_joints_SR_destrorso.keys()):  
            MC = MatchingClass2(sample, dict_joints_SR_destrorso[key])
            MC.weight = 1
            loss = MC.loss()
            dict_losses[key] = loss

        sorted_d = sorted(dict_losses.items(), key=operator.itemgetter(1))
        best_5 = sorted_d[:5]
        worst = sorted_d[-1]

        best_worst_cases_2[i] = [best_5, worst]
    return best_worst_cases_2