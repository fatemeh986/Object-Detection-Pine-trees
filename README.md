# Pine Tree Object Detection MRCNN
In this project Pine Tree (different types) is the main object to detect. For this purpose, the **Matterport/Mask_RCNN** algorith and **COCO** 
pre-trained model h5 (https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) is used. 
The main code of [Mask-RCNN-TF2] (https://github.com/ahmedfgad/Mask-RCNN-TF2) was used. 
The important thing is the **model.py** and **utils.py** is renewd than Matterport/Mask_RCNN because the previous model is not compatible 
with new version of some libraries like Tensorflow and Keras.

## What is changed in this project?
Some small things have changed in this project from the main code that may help others.
### Remember to define a root directory to read data and download models. In the main code it was done by defining ROOT_DIR.
### Try to change the path of train and validation dirctory as your directory names and information.
### Change EPOCH, learning rate and other hyperparameters as you need for your project.
