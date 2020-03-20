import os
import time
import pickle
import cv2

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import threading
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')

from module_pose.pose_openpose_rim import PoseOpenpose
from module_pose.pose_thinpose_rim import PoseThinpose

openpose = PoseOpenpose()
openpose.Setup()

thinpose = PoseThinpose()
thinpose.Setup()

ichannel = grpc.insecure_channel("localhost:8500")
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

# measure_module = "pose_openpose"
measure_module = "pose_thinpose"

image_path = '/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/images/p1.jpg'
frame_id = 0

duration_sum = 0.0

while (frame_id < 10):
  start = time.time()

  if (measure_module == "pose_openpose" or measure_module == "pose_thinpose"):
    image = cv2.imread(image_path)
    request = dict()
    request['client_input'] = image

  if (measure_module == "pose_openpose"):
    module_instance = openpose
  elif (measure_module == "pose_thinpose"):
    module_instance = thinpose

  module_instance.PreProcess(request = request, istub = istub, grpc_flag = False)
  module_instance.Apply()
  next_request = module_instance.PostProcess(grpc_flag = False)

  end = time.time()
  duration = end - start
  print("duration = %s" % str(duration))
  duration_sum += duration

  if (measure_module == "pose_openpose" or measure_module == "pose_thinpose"):
    print(next_request['FINAL'][0].body_parts.values()[0].get_part_name())

  # if (frame_id == 32):
  #   if (measure_module == "pose_openpose"):
  #     pickle_output = "/home/yitao/Downloads/tmp/docker-share/pickle_tmp_combined/tf-pose-estimation/pickle_tmp/pose_openpose/%s" % (str(frame_id).zfill(3))
  #     with open(pickle_output, 'w') as f:
  #        pickle.dump(request, f)
  #   elif (measure_module == "pose_thinpose"):
  #     pickle_output = "/home/yitao/Downloads/tmp/docker-share/pickle_tmp_combined/tf-pose-estimation/pickle_tmp/pose_thinpose/%s" % (str(frame_id).zfill(3))
  #     with open(pickle_output, 'w') as f:
  #        pickle.dump(request, f)

  frame_id += 1

print("On average, it takes %s sec per %s request" % (str(duration_sum / frame_id), measure_module))
