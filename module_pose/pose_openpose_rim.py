import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import cv2
import tensorflow as tf
import pickle

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')
from tf_pose.estimator import PoseEstimator

class PoseOpenpose:

  @staticmethod
  def Setup():
    PoseOpenpose.resize_out_ratio = 4.0

  def PreProcess(self, request, istub, grpc_flag):
    if (grpc_flag):
      self.image = tensor_util.MakeNdarray(request.inputs["client_input"])
    else:
      self.image = request["client_input"]

    self.istub = istub

  def Apply(self):
    upsample_size = [int(self.image.shape[0] / 8 * PoseOpenpose.resize_out_ratio), int(self.image.shape[1] / 8 * PoseOpenpose.resize_out_ratio)]

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'pose_openpose'
    request.model_spec.signature_name = 'predict_images'
    request.inputs['tensor_image'].CopyFrom(
      tf.contrib.util.make_tensor_proto(self.image, shape = [1, 232, 217, 3], dtype=np.float32))
    request.inputs['upsample_size'].CopyFrom(
      tf.contrib.util.make_tensor_proto(upsample_size, shape = [2], dtype=np.int32))

    result = self.istub.Predict(request, 10.0)  # 10 secs timeout

    tensor_heatMat_up = tensor_util.MakeNdarray(result.outputs['tensor_heatMat_up'])
    tensor_pafMat_up = tensor_util.MakeNdarray(result.outputs['tensor_pafMat_up'])
    tensor_peaks = tensor_util.MakeNdarray(result.outputs['tensor_peaks'])

    peaks = tensor_peaks[0]
    heatMat = tensor_heatMat_up[0]
    pafMat = tensor_pafMat_up[0]

    self.humans = PoseEstimator.estimate_paf(peaks, heatMat, pafMat)

  def PostProcess(self, grpc_flag):
    # for human in self.humans:
    #   for k, body_part in human.body_parts.iteritems():
    #     print(body_part)
    #     print(body_part.uidx)
    #     print(body_part.x)
    #     print(body_part.y)
    #     print(body_part.score)
    #     print(body_part.get_part_name())

    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      # next_request.inputs["client_input"].CopyFrom(
      #   tf.make_tensor_proto(self.image))
      next_request.inputs["FINAL"].CopyFrom(
        tf.make_tensor_proto("OK"))
    else:
      next_request = dict()
      # next_request["client_input"] = self.image
      next_request["FINAL"] = self.humans
    return next_request
