import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
import pickle

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')
from module_pose.pose_openpose_rim import PoseOpenpose
from module_pose.pose_recognition_rim import PoseRecognition

import cv2
image_reader = cv2.VideoCapture("/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/images/exercise.avi")

ichannel = grpc.insecure_channel('0.0.0.0:8500')
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

pose = PoseOpenpose()
pose.Setup()

recog = PoseRecognition()
recog.Setup()

simple_route_table = "pose_openpose-pose_recognition"
route_table = simple_route_table

sess_id = "chain_pose-000"

frame_id = 1

while (frame_id < 48):
  print("Processing %dth image" % frame_id)

  _, image = image_reader.read()
  request = dict()
  request["client_input"] = image

  pose.PreProcess(request, istub, False)
  pose.Apply()
  next_request = pose.PostProcess(False)
  # print(next_request["humans"])

  # if (frame_id == 32):
  #   if (True):
  #     pickle_output = "/home/yitao/Downloads/tmp/docker-share/pickle_tmp_combined/tf-pose-estimation/pickle_tmp/pose_openpose/%s" % (str(frame_id).zfill(3))
  #     with open(pickle_output, 'w') as f:
  #        pickle.dump(next_request, f)

  recog.PreProcess(next_request, istub, False)
  recog.Apply()
  next_request = recog.PostProcess(False)

  print(next_request["FINAL"])

  frame_id += 1
