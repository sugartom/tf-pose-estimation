import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')
from module_pose.pose_openpose_tf import PoseOpenpose
from module_pose.pose_recognition_tf import PoseRecognition

import cv2
image_reader = cv2.VideoCapture("/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/images/exercise.avi")
pose = PoseOpenpose()
pose.Setup()

recog = PoseRecognition()
recog.Setup()


frame_id = 1
while (frame_id < 32):
  print("Processing %dth image" % frame_id)

  _, image = image_reader.read()

  pose.PreProcess(image)
  pose.Apply()
  humans = pose.PostProcess()
  # print(humans)

  recog.PreProcess(humans)
  recog.Apply()
  predict_label = recog.PostProcess()

  print(predict_label)

  frame_id += 1