# POSE-ID-on—A Novel Framework for Artwork Pose Clustering

Pose-ID-on is a free and open source pipeline for **pose clustering of human statues**, to gather similar statues based on their poses. It is authored by [Valerio Marsocci](https://github.com/VMarsocci) and [Lorenzo Lastilla](https://github.com/L9L4). The pose-tracking stage of the process is based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

<div align="center"><img src="images/logo_poseidon.png", width="500"></div>

## Contents
1. [Features](#features)
2. [Installation](#installation)
4. [Usage](#usage)
5. [Output](#output)
6. [License](#license)
7. [Citation](#citation)

<div align="center"><img src="images/example.png", width="700"></div>

## Features

This pipeline, given respectively a set of images of statues and of keyponts, gathers two products:
- **pose comparison**, carried out in two different ways: the first method is slower than the second, but both lead to very satisfying results;
- **pose clustering**, based on K-Medians algorithm in a non-euclidean space. Also K-Means and Hierarchical clustering algorithms are implemented.

#### Pose comparison example
The example shows the query pose, the four closest poses and the farthest one.
<div align="center"><img src="images/comparison_ex.png", width="700"></div>

#### Pose clustering example
Following you can find an example of two clusters, the first one from a five clusters clustering and the second one from a ten clusters clustering.
<div align="center"><img src="images/clustering_5_ex1.PNG", width="700"></div>
<div align="center"><img src="images/clustering_10_ex1.PNG", width="700"></div>


## Installation

- Download [Python 3](https://www.python.org/)
- Install the packages:
```bash
pip install -r requirements.txt
```


## Usage

After cloning the repo, the user will find three notebooks:
- *OpenPose_install_Colab.ipynb*: in this notebook, a pipeline to clone the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) repo, through [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb), and to obtain the pose-tracking of the statues, is implemented. This procedure provides a set of rendered images and a set of keypoints. Of course, the images and the coordinates of the keypoints can be provided to the following notebook from any source. Whether the rendered images and keypoints are produced by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) or not, images must be in *.png* format, keypoints in *.json* format. Images must be placed in an *./image/* folder in the root directory, keypoints in a *./keypoints/* folder in the root directory. In conclusion, for a given pose called *pose_example*, two files must be provided to the following notebook:
    - a rendered image *./image/pose_example_rendered.png*
    - a set of keypoints *./keypoints/pose_example_keypoints.json*
- *Matching_and_clustering.ipynb*: in this notebook, just following the instructions accurately provided, the pose comparison and the pose clustering can be gathered. The function that loads the keypoints is optimized for the OpenPose data structure: each set of keypoints is stored in a *.json*, with several subsections. We are interested in ```data['people'][0]['pose_keypoints_2d']```, that indicates the 2D coordinates of the keypoints of the foreground statue. So, if you don't use the OpenPose pipeline, be sure that the data are in this form. Each part of the notebook is accurately explained and optimized.
- *Noise_and_errors.ipynb*, divided into two sections. In the first part, some noise is added to the poses, and some graphs are provided, in order to understand the effect of noise on the comparison. In the second section, the error in the pose reconstruction of the centroids of the clustering is computed.

## Output

The user can gather two types of output:
- directly shown in the notebook;
- saved locally:
  - *.txt* files with the most and least similar poses (with respect to a given one) based on the two comparison methods (we suggest to save these files, that could be successively loaded, without the need of computing them several times). The name of these files can be easily choosen by the user, directly in the notebook;
  - *./Cluster/* directory, created when the clustering algorithm is launched. In this directory, *n* images of reconstructed poses from centroids are saved (with *n* number of clusters), together with a *.txt* file with *n* lists, each containing the name of the images belonging to the *i-th* cluster.

## License

This is an open access article distributed under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/) which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.

## Citation

```bash
@Article{ijgi10040257,
AUTHOR = {Marsocci, Valerio and Lastilla, Lorenzo},
TITLE = {POSE-ID-on—A Novel Framework for Artwork Pose Clustering},
JOURNAL = {ISPRS International Journal of Geo-Information},
VOLUME = {10},
YEAR = {2021},
NUMBER = {4},
ARTICLE-NUMBER = {257},
URL = {https://www.mdpi.com/2220-9964/10/4/257},
ISSN = {2220-9964},
DOI = {10.3390/ijgi10040257}
}
```
