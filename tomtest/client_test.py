# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Send JPEG image to tensorflow_model_server loaded with inception model.
"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import time
import grpc
import tensorflow as tf
from tensorflow.python.framework import tensor_util

import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')
from tf_pose import common
from tf_pose.estimator import PoseEstimator

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('image', '', 'path to image in JPEG format')
FLAGS = tf.app.flags.FLAGS


def main(_):
  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'tf_openpose'
  request.model_spec.signature_name = 'predict_images'

  image_path = '/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/images/p1.jpg'
  resize_out_ratio = 4.0
  
  runNum = 105
  humans = None
  durationSum = 0.0
  runCount = 0

  for i in range(runNum):
    start = time.time()

    image = common.read_imgfile(image_path, None, None)
    upsample_size = [int(image.shape[0] / 8 * resize_out_ratio), int(image.shape[1] / 8 * resize_out_ratio)]

    request.inputs['tensor_image'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image, shape = [1, 232, 217, 3], dtype=np.float32))
    request.inputs['upsample_size'].CopyFrom(
      tf.contrib.util.make_tensor_proto(upsample_size, shape = [2], dtype=np.int32))

    result = stub.Predict(request, 10.0)  # 10 secs timeout
    tensor_heatMat_up = tensor_util.MakeNdarray(result.outputs['tensor_heatMat_up'])
    tensor_pafMat_up = tensor_util.MakeNdarray(result.outputs['tensor_pafMat_up'])
    tensor_peaks = tensor_util.MakeNdarray(result.outputs['tensor_peaks'])

    peaks = tensor_peaks[0]
    heatMat = tensor_heatMat_up[0]
    pafMat = tensor_pafMat_up[0]

    humans = PoseEstimator.estimate_paf(peaks, heatMat, pafMat)

    end = time.time()
    duration = (end - start)
    print("it takes %s sec" % str(duration))

    if (i > 10):
      runCount += 1
      durationSum += duration

  print("On average, it takes %f sec over %d runs." % (durationSum / runCount, runCount))
  print(humans)

if __name__ == '__main__':
  tf.app.run()