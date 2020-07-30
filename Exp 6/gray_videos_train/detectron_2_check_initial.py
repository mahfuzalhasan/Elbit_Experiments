# You may need to restart your runtime prior to this, to let your installation take effect

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog




#im = cv2.imread("./04219.png")
#print(im.shape)

def get_model():

    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.MODEL.WEIGHTS = "/home/c3-0/mahfuz/Elbit_curriculum_learning/gray_model_sent_to_elbit/model_final.pth"
    cfg.MODEL.WEIGHTS = "./model_final.pth"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 100  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 23  
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   
    predictor = DefaultPredictor(cfg)
    #outputs = predictor(im)
    return predictor

'''
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2.imwrite('input_1.jpg', v.get_image()[:, :, ::-1])

'''
