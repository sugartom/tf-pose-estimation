import threading
import cv2
import grpc
import time
import numpy as np
import os
import pickle
import sys

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow_serving.apis import prediction_service_pb2_grpc

sys.path.append('/home/yitao/Documents/edge/D2-system/')
from utils_d2 import misc
from modules_d2.video_reader import VideoReader

sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')

from module_pose.pose_openpose_d2 import PoseOpenpose
from module_pose.pose_recognition_d2 import PoseRecognition
from module_pose.tracker_d2 import PoseTracker

openpose = PoseOpenpose()
openpose.Setup()

recognition = PoseRecognition()
recognition.Setup()

MAX_MESSAGE_LENGTH = 1024 * 1024 * 256
options = [('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_message_length', MAX_MESSAGE_LENGTH)]
ichannel = grpc.insecure_channel("localhost:8500", options = options)
istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

module_name = "pose_recognition"

pickle_directory = "%s/pickle_d2/tf-pose-estimation/%s" % (os.environ['RIM_DOCKER_SHARE'], module_name)
if not os.path.exists(pickle_directory):
  os.makedirs(pickle_directory)

batch_size = 1
parallel_level = 1
run_num = 10

def runBatch(batch_size, run_num, tid):
  start = time.time()

  pose_reader = VideoReader()
  pose_reader.Setup("%s/images/exercise.avi" % os.environ['POSE_ESTIMATION'])

  multiperson_tracker = PoseTracker()
  my_lock = threading.Lock()

  frame_id = 0
  batch_id = 0

  while (batch_id < run_num):
    module_instance = misc.prepareModuleInstance(module_name)
    data_array = []

    if (module_name == "pose_openpose"):
      for i in range(batch_size):
        client_input = misc.getClientInput("chain_pose", pose_reader)
        request = dict()
        request["client_input"] = client_input
        data_dict = module_instance.GetDataDict(request, grpc_flag = False)
        data_array.append(data_dict)
        frame_id += 1
    elif (module_name == "pose_recognition"):
      pickle_input = "%s/%s" % ("%s/pickle_d2/%s/%s" % (os.environ['RIM_DOCKER_SHARE'], "tf-pose-estimation", "pose_openpose"), str(frame_id + 1).zfill(3))
      with open(pickle_input) as f:
        request = pickle.load(f)
        data_dict = module_instance.GetDataDict(request, grpc_flag = False)
        data_array.append(data_dict)
      frame_id += 1

    batched_data_dict = module_instance.GetBatchedDataDict(data_array, batch_size)

    if (module_name == "pose_recognition"):
      batched_result_dict = module_instance.Apply(batched_data_dict, batch_size, istub, multiperson_tracker, my_lock)
    else:
      batched_result_dict = module_instance.Apply(batched_data_dict, batch_size, istub)

    batched_result_array = module_instance.GetBatchedResultArray(batched_result_dict, batch_size)

    for i in range(len(batched_result_array)):
      # deal with the outputs of the ith input in the batch
      result_dict = batched_result_array[i]

      # each input might have more than one outputs
      result_list = module_instance.GetResultList(result_dict)

      for result in result_list:
        next_request = module_instance.GetNextRequest(result, grpc_flag = False)

        # if (module_name == "pose_openpose"):
        #   # print(len(next_request["humans"]))
        #   # print(next_request["humans"])

        #   pickle_output = "%s/%s" % (pickle_directory, str(frame_id).zfill(3))
        #   with open(pickle_output, 'w') as f:
        #     pickle.dump(next_request, f)

        if (module_name == "pose_recognition"):
          print(next_request["output_label"])

    batch_id += 1

  end = time.time()
  print("[Thread-%d] it takes %.3f sec to run %d batches of batch size %d" % (tid, end - start, run_num, batch_size))


# ========================================================================================================================

start = time.time()

thread_pool = []
for i in range(parallel_level):
  t = threading.Thread(target = runBatch, args = (batch_size, run_num, i))
  thread_pool.append(t)
  t.start()

for t in thread_pool:
  t.join()

end = time.time()
print("overall time = %.3f sec" % (end - start))
