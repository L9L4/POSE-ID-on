import matplotlib.pyplot as plt
import cv2
import os

def plot_images(dir_im, t = 0, best_worst_cases = {}):
  fig = plt.figure(figsize=(40, 40))
  im_list = [a[0] for a in best_worst_cases[list(best_worst_cases.keys())[t]][0]] + [best_worst_cases[list(best_worst_cases.keys())[t]][1][0]]
  
  columns = 6
  rows = 1
  for i in range(1, columns*rows +1):
    im = cv2.imread(dir_im + '/'+im_list[i-1] + '_rendered.png')
    im = cv2.resize(im, (200,400))
    ax = fig.add_subplot(rows, columns, i)
    plt.imshow(im[:,:,::-1])
    #plt.axis('off')
    ax.tick_params(labelbottom=False, bottom = False, labelleft = False, left = False)
    if i == 1:
      plt.title("Query", fontsize= 14)
      ax.set_xlabel(im_list[i-1], fontsize= 13)
    elif i > 1 and i < columns*rows:
      plt.title("Closest result " + str(i-1), fontsize= 14)
      ax.set_xlabel(im_list[i-1], fontsize= 13)
    else:
      plt.title("Farthest result " +  str(1), fontsize= 14)
      ax.set_xlabel(im_list[i-1], fontsize= 13)
  
  plt.show()
  #fig.savefig("prova.png")

  print("Query: ",im_list[0],  '\n')
  print("---------------\n")
  print("Closest results: \n")
  for i in range(1,5):
    print(im_list[i], '\n')
  print("---------------\n")
  print("Farthest result: ", im_list[5])


def save_images(dir_im, t = 0, method = 1, dictionary = {}):
  
  assert method == 1 or method == 2, "Invalid method"

  dir_res = os.path.join(os.getcwd(), "Results_method_" + str(method))
  if not os.path.exists(dir_res):
    os.makedirs(dir_res)
  

  fig = plt.figure(figsize=(40, 40))
  im_list = [a[0] for a in dictionary[list(dictionary.keys())[t]][0]] + [dictionary[list(dictionary.keys())[t]][1][0]]

  columns = 6
  rows = 1
  for i in range(1, columns*rows +1):
    im = cv2.imread(dir_im + '/'+im_list[i-1] + '_rendered.png')
    im = cv2.resize(im, (200,400))
    ax = fig.add_subplot(rows, columns, i)
    plt.imshow(im[:,:,::-1])
    #plt.axis('off')
    ax.tick_params(labelbottom=False, bottom = False, labelleft = False, left = False)
    if i == 1:
      plt.title("Query", fontsize= 14)
      ax.set_xlabel(im_list[i-1], fontsize= 13)
    elif i > 1 and i < columns*rows:
      plt.title("Closest result " + str(i-1), fontsize= 14)
      ax.set_xlabel(im_list[i-1], fontsize= 13)
    else:
      plt.title("Farthest result " +  str(1), fontsize= 14)
      ax.set_xlabel(im_list[i-1], fontsize= 13)

  #plt.show()
  plt.savefig(dir_res+"/{}.png".format(im_list[0]))
  plt.close()


def show_pose(i, dict_joints, dir_im):
    punti_prova = dict_joints[list(dict_joints.keys())[i]]
    fig, ax = plt.subplots(figsize = (15,15))
    im = cv2.imread(dir_im+'/'+list(dict_joints.keys())[i] + "_rendered.png")
    print(os.listdir(dir_im)[i])
    plt.imshow(im/255.0)
    for n in range(len(punti_prova)):
        plt.plot(punti_prova[n][0], punti_prova[n][1], 'ro')
        ax.annotate(n, (punti_prova[n][0], punti_prova[n][1]))
    return

def show_single_pose(i, punti_prova, dict_joints, dir_im):
    fig, ax = plt.subplots(figsize = (15,15))
    im = cv2.imread(dir_im+'/'+list(dict_joints.keys())[i] + "_rendered.png")
    print(os.listdir(dir_im)[i])
    plt.imshow(im/255.0)
    for n in range(len(punti_prova)):
        plt.plot(punti_prova[n][0], punti_prova[n][1], 'ro')
        ax.annotate(n, (punti_prova[n][0], punti_prova[n][1]))
    return

def watch_samples(n_cl_show, df_cls, rec_poses, n_cluster_list, dir_im):
    links = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[1,8],[7,6],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14]]

    all_samples = {}
    for n in n_cluster_list:
        samples = []
        df_clustering = df_cls[n]
        for i in range(n):
            a = df_clustering[df_clustering['label'] == i]
            samples.append(list(a['label'].sample(min(5, len(a))).index))
        all_samples[n] = samples
        
    for j in range(n_cl_show):
        rec_pose = rec_poses[n_cl_show][j]
        im_list = all_samples[n_cl_show][j]
        fig = plt.figure(figsize=(40, 40))
        columns = min(5, len(im_list))
        rows = 1
        for i in range(1, columns*rows +1):
            im = cv2.imread(dir_im + '/'+im_list[i-1] + '_rendered.png')
            im = cv2.resize(im, (200,400))
            ax = fig.add_subplot(rows, columns, i)
            #plt.axis('off')
            ax.tick_params(labelbottom=False, bottom = False, labelleft = False, left = False)
            if i == 1:
                for n in range(len(rec_pose)):
                    plt.plot(rec_pose[n][0], rec_pose[n][1], 'ro')
                    ax.annotate(n, (rec_pose[n][0], rec_pose[n][1]))
                    ax.set_aspect(aspect = "equal")   
                for l in range(len(links)):
                    p1, p2 = links[l]
                    plt.plot([rec_pose[p1][0], rec_pose[p2][0]],[rec_pose[p1][1], rec_pose[p2][1]], '-')
            else:
              plt.imshow(im[:,:,::-1])
              plt.title("Random example " + str(i-1), fontsize= 14)
              ax.set_xlabel(im_list[i-1], fontsize= 13)
        plt.show()
    return