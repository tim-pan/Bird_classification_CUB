
# VRDL_HW1
---
for TA: if you want to reproduce the submission, please just jump to the last part.
---
## task
bird category classification([CUB dataset 2010](http://www.vision.caltech.edu/visipedia/CUB-200.html))
## Environment
pytorch
colab GPU(Tesla P100 or Persistence-M)
## requirement
1. your GPU RAM is larger than 25gb </br>
2. pytorch with cuda version
## introduction
there are 
- 3 py files
- 6 ipynb files
- 2 folders

**1. best_train.ipynb:**
this file has my best try hyperparameter settings, if you want to reproduce whole **training** procedure, please run this file.

**2. birdloader.py**
data preprocessing file

**3. data_split.ipynb**
before data preprocessing, use this file to split training data into training set and dev(validation) set

**4. evalute.ipynb**
reproduce the experiment outcome, will generate a answer.txt file

**5. experiment_vit.ipynb**
experiment about ViT

**6. experiment_ensemble.ipynb**
experiment about ensembles of some models

**7. eperiment_single_cnn.ipynb**
experiment about tuning hyperparameters for different CNN

**8. function.py**
some function that this homework should use, such as train, eval, etc...

**9. model-para**
this is a folder, there are three model weight dictionaries in this folder

**10. dataset**
dataset folder.

**11. inference.py**
the file used to reproduce the submission

## reproduce the submission
1. refer to the following link, download the folder:</br>
https://drive.google.com/file/d/1D38aXXP2ELYctE2OqweDZPi0jHjc1Rvo/view?usp=sharing</br>
dataset and py.files already in this link, you don't need to download it again.
2. on cmd, cd to the folder you downloaded
3. run `inference.py`  your GPU


