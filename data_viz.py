import matplotlib.pyplot as plt
import cv2

def plot_images(t = 0, dictionary = {}):
  #dir_joints = '/content/content/Shared drives/Pose-tracking/Scraping/joint'
  #dir_im = '/content/content/Shared drives/Pose-tracking/Scraping/immagini_testate_e_joint'
  fig = plt.figure(figsize=(40, 40))
  # ocho
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


def save_images(t = 0, method = 0, dictionary = {}, j=0):
  
  assert method == 0 or method == 1 or method == 2, "Invalid method"
  
  dir_joints = '/content/content/Shared drives/Pose-tracking/preselezione sox/joints/selezione'
  dir_im = '/content/content/Shared drives/Pose-tracking/preselezione sox/Poses_and_images/selezione'
  #dir_im = '/content/content/Shared drives/Pose-tracking/preselezione sox/sub-selezione_lollo/sel'
  if method == 0:
    dir_res = '/content/content/Shared drives/Pose-tracking/risultati/risultati_metodo1/'
  elif method == 1:
    dir_res = '/content/content/Shared drives/Pose-tracking/risultati/risultati_metodo2/'
  elif method == 2:
    dir_res = '/content/content/Shared drives/Pose-tracking/risultati/risultati_metodo_comb/'
  

  fig = plt.figure(figsize=(30, 15))
  im_list = [t] + [int(dictionary[t][0][i][0]) for i in range(len(dictionary[t][0]))] + [dictionary[t][1][0]]

  columns = 7
  rows = 1
  for i in range(1, columns*rows +1):
    im = cv2.imread(dir_im + '/'+os.listdir(dir_im)[im_list[i-1]])
    im = cv2.resize(im, (200,400))
    ax = fig.add_subplot(rows, columns, i)
    plt.imshow(im/255.0)
    #plt.axis('off')
    ax.tick_params(labelbottom=False, bottom = False, labelleft = False, left = False)
    if i == 1:
      plt.title("Query", fontsize= 14)
      ax.set_xlabel(os.listdir(dir_im)[im_list[0]].split('.')[0], fontsize= 13)
    elif i > 1 and i < columns*rows:
      plt.title("Closest result " + str(i-1), fontsize= 14)
      ax.set_xlabel(os.listdir(dir_im)[im_list[i-1]].split('.')[0], fontsize= 13)
    else:
      plt.title("Farthest result " +  str(1), fontsize= 14)
      ax.set_xlabel(os.listdir(dir_im)[im_list[i-1]].split('.')[0], fontsize= 13)
  
  #plt.show()
  fig.savefig(dir_res+"{}.png".format(j))
  plt.close()