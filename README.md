# Pine Tree Object Detection_MRCNN
In this project, the goal is to detect different types of pine trees. To achieve this, the **Matterport/Mask_RCNN** algorithm and **COCO** 
pre-trained model h5 (https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) are used. 
The primary codebase utilized is from [Mask-RCNN-TF2] (https://github.com/ahmedfgad/Mask-RCNN-TF2). 
Notably, the **model.py** and **utils.py** files have been updated from the Matterport/Mask_RCNN version to ensure compatibility with newer versions of libraries like TensorFlow and Keras.

## Changes Made in This Project
### Key Modifications:
1. Defined a root directory for reading data and downloading models. In the original code, this was handled by setting ROOT_DIR.
2. Adjust the paths for the training and validation directories to match your specific directory structure.
3. Customized hyperparameters such as the number of epochs and learning rate to suit the needs of your project.
