from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')
from tf_pose.estimator import PoseEstimator

import cv2
import pickle
import numpy as np

class PoseOpenpose:

  # initialize static variable here
  @staticmethod
  def Setup():
    PoseOpenpose.resize_out_ratio = 4.0

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request, grpc_flag):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    if (grpc_flag):
      image = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      image = request["client_input"]

    image = cv2.resize(image, (217, 232))
    upsample_size = [int(image.shape[0] / 8 * PoseOpenpose.resize_out_ratio), int(image.shape[1] / 8 * PoseOpenpose.resize_out_ratio)]

    data_dict["client_input"] = image
    data_dict["upsample_size"] = upsample_size

    return data_dict

  # for an array of requests from a batch, convert them to a dict,
  # where each key has a lit of values
  # input: data_array = [{"image": image1, "meta": meta1}, {"image": image2, "meta": meta2}]
  # output: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  def GetBatchedDataDict(self, data_array, batch_size):
    if (len(data_array) != batch_size):
      print("[Error] GetBatchedDataDict() batch size not matched...")
      return None
    else:
      batched_data_dict = dict()

      # for each key in data_array[0], convert it to batched_data_dict[key][]
      if (batch_size == 1):
        batched_data_dict["client_input"] = [data_array[0]["client_input"]]
      else:
        batched_data_dict["client_input"] = data_array[0]["client_input"]
        for data in data_array[1:]:
          batched_data_dict["client_input"] = np.append(batched_data_dict["client_input"], data["client_input"], axis = 0)
      
      batched_data_dict["upsample_size"] = []
      for data in data_array:
        batched_data_dict["upsample_size"].append(data["upsample_size"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      request = predict_pb2.PredictRequest()
      request.model_spec.name = 'pose_openpose'
      request.model_spec.signature_name = 'predict_images'
      request.inputs['tensor_image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batched_data_dict["client_input"][0], shape = [1, 232, 217, 3], dtype=np.float32))
      request.inputs['upsample_size'].CopyFrom(
        tf.contrib.util.make_tensor_proto(batched_data_dict["upsample_size"][0], shape = [2], dtype=np.int32))

      result = istub.Predict(request, 10.0)  # 10 secs timeout

      tensor_heatMat_up = tensor_util.MakeNdarray(result.outputs['tensor_heatMat_up'])
      tensor_pafMat_up = tensor_util.MakeNdarray(result.outputs['tensor_pafMat_up'])
      tensor_peaks = tensor_util.MakeNdarray(result.outputs['tensor_peaks'])

      peaks = tensor_peaks[0]
      heatMat = tensor_heatMat_up[0]
      pafMat = tensor_pafMat_up[0]

      humans = PoseEstimator.estimate_paf(peaks, heatMat, pafMat)

      batched_result_dict["humans"] = [humans]

      return batched_result_dict

  # input: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  # output: batched_result_array = [{"bounding_boxes": [bb1_in_image1, bb2_in_image1]}, {"bounding_boxes": [bb1_in_image2]}]
  def GetBatchedResultArray(self, batched_result_dict, batch_size):
    if (batch_size != len(batched_result_dict[batched_result_dict.keys()[0]])):
      print("[Error] GetBatchedResultArray() batch size not matched...")
      return None
    else:
      batched_result_array = []

      for i in range(batch_size):
        my_dict = dict()
        my_dict["humans"] = batched_result_dict["humans"][i]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []
    # for i in range(len(result_dict[result_dict.keys()[0]])):
    result_list.append({"humans": result_dict["humans"]})

    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["humans"].CopyFrom(
        tf.make_tensor_proto(pickle.dumps(result["humans"])))
    else:
      next_request = dict()
      next_request["humans"] = result["humans"]
    return next_request
