from tensorflow_serving.apis import predict_pb2
from tensorflow.python.framework import tensor_util
import tensorflow as tf

# from tracker_d2 import PoseTracker
from multi_classifier import MultiPersonClassifier

import pickle

class PoseRecognition:

  def remove_skeletons_with_few_joints(self, skeletons):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    for skeleton in skeletons:
      px = skeleton[2:2+13*2:2]
      py = skeleton[3:2+13*2:2]
      num_valid_joints = len([x for x in px if x != 0])
      num_leg_joints = len([x for x in px[-6:] if x != 0])
      total_size = max(py) - min(py)
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
        # add this skeleton only when all requirements are satisfied
        good_skeletons.append(skeleton)
    return good_skeletons

  def humans_to_skels_list(self, humans):
    # if scale_h is None:
    scale_h = PoseRecognition._scale_h
    skeletons = []
    NaN = 0
    for human in humans:
      skeleton = [NaN]*(18*2)
      for i, body_part in human.body_parts.items(): # iterate dict
        idx = body_part.part_idx
        skeleton[2*idx]=body_part.x
        skeleton[2*idx+1]=body_part.y * scale_h
      skeletons.append(skeleton)
    return skeletons, scale_h

  # initialize static variable here
  @staticmethod
  def Setup():
    PoseRecognition._scale_h = 1.0 * 288 / 384
    # PoseRecognition.multiperson_tracker = PoseTracker()
    PoseRecognition.multiperson_classifier = MultiPersonClassifier("/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/exported_models/pose_recognition/trained_classifier.pickle", ['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave'])

  # convert predict_pb2.PredictRequest()'s content to data_dict
  # input: request["image"] = image
  #        request["meta"] = meta
  # output: data_dict["image"] = image
  #         data_dict["meta"] = meta
  def GetDataDict(self, request, grpc_flag):
    data_dict = dict()

    # do the conversion for each key in predict_pb2.PredictRequest()
    # if (grpc_flag):
    #   image = tensor_util.MakeNdarray(request.inputs["client_input"])
    # else:
    #   image = request["client_input"]
    # data_dict["client_input"] = image

    if (grpc_flag):
      humans = pickle.loads(str(tensor_util.MakeNdarray(request.inputs["humans"])))
    else:
      humans = request["humans"]

    data_dict["humans"] = humans

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
      # if (batch_size == 1):
      #   batched_data_dict["client_input"] = [data_array[0]["client_input"]]
      # else:
      #   batched_data_dict["client_input"] = data_array[0]["client_input"]
      #   for data in data_array[1:]:
      #     batched_data_dict["client_input"] = np.append(batched_data_dict["client_input"], data["client_input"], axis = 0)

      batched_data_dict["humans"] = []
      for data in data_array:
        batched_data_dict["humans"].append(data["humans"])

      return batched_data_dict

  # input: batched_data_dict = {"image": [image1, image2], "meta": [meta1, meta2]}
  # output: batched_result_dict = {"bounding_boxes": [[bb1_in_image1, bb2_in_image1], [bb1_in_image2]]}
  def Apply(self, batched_data_dict, batch_size, istub, multiperson_tracker, my_lock):
    if (batch_size != len(batched_data_dict[batched_data_dict.keys()[0]])):
      print("[Error] Apply() batch size not matched...")
      return None
    else:
      batched_result_dict = dict()

      skeletons, scale_h = self.humans_to_skels_list(batched_data_dict["humans"][0])
      skeletons = self.remove_skeletons_with_few_joints(skeletons)

      my_lock.acquire()
      dict_id2skeleton = multiperson_tracker.track(skeletons)
      my_lock.release()

      output_label = ""

      if (len(dict_id2skeleton)):
        dict_id2label = PoseRecognition.multiperson_classifier.classify(dict_id2skeleton)
        min_id = min(dict_id2skeleton.keys())
        output_label = dict_id2label[min_id]

      batched_result_dict["output_label"] = [output_label]

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
        my_dict["output_label"] = batched_result_dict["output_label"][i]
        batched_result_array.append(my_dict)

      return batched_result_array

  # input: result_dict = {"bounding_boxes": [bb1_in_image1, bb2_in_image1]}
  # output: result_list = [{"bounding_boxes": bb1_in_image1}, {"bounding_boxes": bb2_in_image1}]
  def GetResultList(self, result_dict):
    result_list = []
    # for i in range(len(result_dict[result_dict.keys()[0]])):
    result_list.append({"output_label": result_dict["output_label"]})

    return result_list

  # input: result = {"bounding_boxes": bb1_in_image1}
  # output: next_request["boudning_boxes"] = bb1_in_image1
  def GetNextRequest(self, result, grpc_flag):
    if (grpc_flag):
      next_request = predict_pb2.PredictRequest()
      next_request.inputs["output_label"].CopyFrom(
        tf.make_tensor_proto(result["output_label"]))
    else:
      next_request = dict()
      next_request["output_label"] = result["output_label"]
    return next_request
