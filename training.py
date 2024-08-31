import os
import sys
import json
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model
from mrcnn import utils

ROOT_DIR = os.path.abspath("C:/Users/fkara/OneDrive/Documents/object detection")
sys.path.append(ROOT_DIR)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
class PineTreeDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "pinetree")

        # images_dir = dataset_dir + '/images/'
        # annotations_dir = dataset_dir + '/annots/'
        images_dir = os.path.join(dataset_dir, 'images')
        annotations_dir = os.path.join(dataset_dir, 'annots')


        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            """ if is_train and int(image_id) >= 150:
                continue

            if not is_train and int(image_id) < 150:
                continue """

            # img_path = images_dir + filename
            # ann_path = annotations_dir + image_id + '.json'
            img_path = os.path.join(images_dir, filename)
            ann_filename = image_id + '.json'
            ann_path = os.path.join(annotations_dir, ann_filename)


            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('pinetree'))
        return masks, asarray(class_ids, dtype='int32')


    def extract_boxes(self, filename):
        with open(filename) as f:
            data = json.load(f)

        boxes = list()
        annotations = data.get('annotations', [])
        for annotation in annotations:
            bbox = annotation.get('bbox')
            if bbox:
                xmin, ymin, width, height = bbox
                xmax = xmin + width
                ymax = ymin + height
                coors = [xmin, ymin, xmax, ymax]
                boxes.append(coors)

        width = data['image']['width']
        height = data['image']['height']
        return boxes, width, height




class PineConfig(mrcnn.config.Config):
    NAME = "pine_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 2

    STEPS_PER_EPOCH = 100

# Train
train_dataset = PineTreeDataset()
train_dataset.load_dataset(dataset_dir=os.path.join(ROOT_DIR, "train"), is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = PineTreeDataset()
validation_dataset.load_dataset(dataset_dir=os.path.join(ROOT_DIR, "train"), is_train=False)
validation_dataset.prepare()

# Model Configuration
pine_config = PineConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir=MODEL_DIR, 
                             config=pine_config)

model.load_weights(COCO_MODEL_PATH, 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=pine_config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')


model_path = os.path.join(ROOT_DIR, "logs")
model.keras_model.save_weights(model_path)