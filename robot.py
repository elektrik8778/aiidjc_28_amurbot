import numpy as np
import pybullet as p
import pybullet_data
import cv2
import tkinter as tk
import threading


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

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)



an1 = 0
an2 = 0
an3 = 0
an4 = 0
an5 = 0
an6 = 0

def gui():
   
    def set_an1(v):
        global an1        
        an1 = v
        
    def set_an2(v):
        global an2        
        an2 = v
    def set_an3(v):
        global an3        
        an3 = v

    def set_an4(v):
        global an4        
        an4 = v

    def set_an5(v):
        global an5        
        an5 = v
    def set_an6(v):
        global an6        
        an6 = v

    master = tk.Tk()
    w1 = tk.Scale(master, from_=-3.2, to=3.2, tickinterval=0.1, resolution=0.1, command = set_an1)
    
    
    
    w2 = tk.Scale(master, from_=-3.2, to=3.2,tickinterval=0.1, resolution=0.1,  command = set_an2)
    

    w3 = tk.Scale(master, from_=-3.2, to=3.2,tickinterval=0.1, resolution=0.1,  command = set_an3)
    w4 = tk.Scale(master, from_=-3.2, to=3.2,tickinterval=0.1, resolution=0.1,  command = set_an4)
    w5 = tk.Scale(master, from_=-3.2, to=3.2,tickinterval=0.1, resolution=0.1,  command = set_an5)
    w6 = tk.Scale(master, from_=-3.2, to=3.2,tickinterval=0.1, resolution=0.1,  command = set_an6)
    w1.pack(side=tk.LEFT)
    w2.pack(side=tk.LEFT)
    w3.pack(side=tk.LEFT)
    w4.pack(side=tk.LEFT)
    w5.pack(side=tk.LEFT)
    w6.pack(side=tk.LEFT)
    master.mainloop()

    
ui = threading.Thread(target=gui)
ui.start()

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(0.001)

p.setRealTimeSimulation(0)

# Add plane
plane_id = p.loadURDF("plane.urdf")



# Add kuka bot
start_pos_ku = [0, 0, 0.001]

start_pos_ur = [-3, 0, 0.001]
start_pos_wsg = [2, 0, 0.001]

start_orientation = p.getQuaternionFromEuler([0, 0, 0])

#grab = p.loadURDF("schunk_wsg50_model-master/models/wsg50_110.urdf", start_pos_wsg, start_orientation)
#kuka_id = p.loadURDF("kuka_experimental/kuka_kr210_support/urdf/kr210l150.urdf",
#                          [0, 0, 0], useFixedBase=1)

kuka_id = p.loadURDF("kuka_iiwa/model.urdf", start_pos_ku, start_orientation)

ur10_id = p.loadURDF("robot_movement_interface-master/dependencies/ur_description/urdf/ur10_robot.urdf", start_pos_ur, start_orientation)

#ur_id = p.loadURDF("ur10_robot.urdf", start_pos_ur, start_orientation)

world_position, world_orientation = p.getLinkState(ur10_id, 6)[:2]
print(world_position)
fov, aspect, nearplane, farplane = 60, 1.0, 0.01, 100


projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)


def kuka_camera():
    # Center of mass position and orientation (of link-7)
    com_p, com_o, _, _, _, _ = p.getLinkState(ur10_id, 6, computeForwardKinematics=True)
    rot_matrix = p.getMatrixFromQuaternion(com_o)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (0, 0, 1) # z-axis
    init_up_vector = (0, 1, 0) # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix = p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
    img = p.getCameraImage(300, 200, view_matrix, projection_matrix)
    image = np.array(img[2])
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # outputs = predictor(img_rgb)
    # v = Visualizer(img_rgb[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu")) #
    # cv2.imshow("camera", out.get_image()[:, :, ::-1])
    
    
    cv2.imshow('f', img_rgb)
    cv2.waitKey(1)
    return img


p.setJointMotorControlArray(
    kuka_id, [2],
    p.POSITION_CONTROL,
    targetPositions=[1.5])
    
# p.setJointMotorControlArray(
#     ur10_id, range(6),
#     p.POSITION_CONTROL,
#     targetPositions=[0, -0.4,-1,-0.3,0,0])

for idx_joint in range(p.getNumJoints(ur10_id)):
    print(p.getJointInfo(ur10_id, idx_joint))

# Main loop
while True:
    #x = list(map(float, input("[0,0,0,0,0,0]").split()))
    #for _ in range(100):
    #    p.stepSimulation()
    #    kuka_camera()
    x = [0,float(an1),float(an2),float(an3),float(an4) , float(an5), float(an6) ]
    
    p.setJointMotorControlArray(ur10_id, range(7),p.POSITION_CONTROL,targetPositions=x)
    
    p.stepSimulation()
    kuka_camera()
   
