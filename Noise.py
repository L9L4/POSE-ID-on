
import numpy as np
import matplotlib.pyplot as plt
from First_method import *
from Second_method import *


class Noises():

  sigma_range = np.logspace(-3, 0, 10)
  n = 10

  def __init__(self, joints, place, method):
    self.joints = joints
    self.place = place
    self.posa = self.joints[self.place]
    self.method = method

  def calc_radius(self):
    punto_medio = []
    j_0 = 0
    j_1 = 0
    for i in range(len(self.posa)):
      j_0 += self.posa[i][0]
      j_1 += self.posa[i][1]
    punto_medio.append([j_0, j_1])
    punto_medio = [punto_medio[0][0]/len(self.posa), punto_medio[0][1]/len(self.posa)]

    dista = 0
    for punto in self.posa:
      dista += np.power(punto[0]-punto_medio[0], 2) + np.power(punto[1]-punto_medio[1], 2)
    dista = dista/len(self.posa)

    return np.sqrt(dista)

  def make_noised_poses(self, sigma = 0.05):
    radius = self.calc_radius()
    noised_poses = []
    for i in range(self.n):
      err = np.random.normal(0, sigma, 30)*radius
      noised = []
      for i in range(len(self.posa)):
        n_j_0 = self.posa[i][0] + err[i]
        n_j_1 = self.posa[i][1] + err[i+15]
        noised.append([n_j_0, n_j_1])
      noised_poses.append(noised)
    return noised_poses

  def noised_poses_sigmas(self):
    dict_noises = {}
    i = 0
    #radius = self.calc_radius()
    for num in self.sigma_range:
      noised_poses = self.make_noised_poses(num)
      dict_noises[i] = noised_poses
      #dict_noises[num] = noised_poses
      i += 1
    return dict_noises
  
  def calc_losses_sigma(self):

    losses_sigma = {}

    dict_noises = self.noised_poses_sigmas()

    for j in range(len(dict_noises.keys())):

      dict_noised = {}
      for i in range(len(dict_noises[j])):
        dict_noised[i] = dict_noises[j][i]

      dict_losses = {}

      for key in dict_noised.keys():
        if self.method == 1:
          MC = MatchingClass1(self.posa, dict_noised[key])
          loss = MC.minimum()[1]
          dict_losses[key] = loss
        elif self.method == 2:
          MC = MatchingClass2(self.posa, dict_noised[key])
          MC.weight = 1
          loss = MC.loss()
          dict_losses[key] = loss
        else:
          raise Exception('Valid methods 1 or 2')

      val = np.mean(list(dict_losses.values()))

      losses_sigma[j] = val
    return losses_sigma

  def print_position(self):
    met = []

    dict_losses_with_nan = {}
  
    dict_joints = {}
    for i in range(len(self.joints)):
      dict_joints[i] = self.joints[i]

    for key in dict_joints.keys():
      if self.method == 1:
        MC = MatchingClass1(self.posa, dict_joints[key])
        loss = MC.minimum()[1]
        dict_losses_with_nan[key] = loss
      elif self.method == 2:
        MC = MatchingClass2(self.posa, dict_joints[key])
        MC.weight = 1
        loss = MC.loss()
        dict_losses_with_nan[key] = loss
      else:
        raise Exception('Valid methods 1 or 2')
      
      dict_losses = {key:val for key, val in dict_losses_with_nan.items() if not isnan(val)}
      
      met.append(dict_losses)

    losses_sigma = self.calc_losses_sigma()
    position = []
    num_position = []

    for t in range(len(losses_sigma)):
      met_con_noise = []
      dict_losses_noisy = dict_losses.copy()

      dict_losses_noisy[list(dict_losses.keys())[-1] + 1] = losses_sigma[t]

      met_con_noise.append(dict_losses_noisy)

      for i in range(len(met_con_noise)):
        sorted_d = sorted(met_con_noise[i].items(), key=operator.itemgetter(1))

      for j in range(len(sorted_d)):
        if sorted_d[j][0] == (list(dict_losses.keys())[-1] + 1):
          position.append(str(j + 1) + '/' + str(len(sorted_d)))
          num_position.append(j+1)
        else:
          continue

    return position, num_position


def graph_losses(losses_sigma, bawc, sigma_range):
  plt.figure(figsize=(10,10))
  plt.plot(sigma_range, list(losses_sigma.values()), 'bo-')
  plt.hlines(bawc, min(sigma_range)-1, max(sigma_range)+1, 'r')
  plt.grid(which = "both")
  plt.title("Noised losses")
  plt.xlabel("Sigma (-)")
  plt.ylabel("Loss (rad)")
  plt.xscale('log')
  plt.yscale('log')
  plt.show()
  plt.close()


def graph_position(position, max_pos, sigma_range):
  plt.figure(figsize=(10,10))
  plt.plot(sigma_range, position, 'bo-')
  plt.hlines(max_pos, min(sigma_range)-1, max(sigma_range)+1, 'r')
  plt.grid(which = "both")
  plt.title("Noised position")
  plt.xlabel("Sigma (-)")
  plt.ylabel("Position (-)")
  plt.xscale('log')
  plt.show()
  plt.close()