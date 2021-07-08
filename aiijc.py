import torch
assert torch.__version__.startswith("1.7")
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300) # Ширина кадров в видеопотоке.
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200) # Высота кадров в видеопотоке.
# wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
#im = cv2.imread("000000439715.jpg")
# cv2.imshow('frame',im)


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

while True:
    ret, img = cap.read()
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu")) #
    cv2.imshow("camera", out.get_image()[:, :, ::-1])
    if cv2.waitKey(10) == 27: # Клавиша Esc
        break
# 

# 
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow('frame',out.get_image()[:, :, ::-1])

# cv2.waitKey(0) 
cap.release()
cv2.destroyAllWindows()

# python3 demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#   --input /home/politehxx/robot_movement_interface/000000439715.jpg \
#   --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl \
#   MODEL.DEVICE cpu
