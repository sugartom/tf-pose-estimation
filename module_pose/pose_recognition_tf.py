from tracker import Tracker
from multi_classifier import MultiPersonClassifier

# OPENPOSE_MODEL = "mobilenet_thin"
# OPENPOSE_IMG_SIZE = "656x368"subl 

# OPENPOSE_MODEL = "cmu"
# OPENPOSE_IMG_SIZE = "432x368"



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

  @staticmethod
  def Setup():
    PoseRecognition._scale_h = 1.0 * 288 / 384
    PoseRecognition.multiperson_tracker = Tracker()
    PoseRecognition.multiperson_classifier = MultiPersonClassifier("/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/exported_models/pose_recognition/trained_classifier.pickle", ['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave'])

  def PreProcess(self, input):
    self.humans = input

  def Apply(self):
    skeletons, scale_h = self.humans_to_skels_list(self.humans)
    skeletons = self.remove_skeletons_with_few_joints(skeletons)
    dict_id2skeleton = PoseRecognition.multiperson_tracker.track(skeletons)

    self.output = ""
    if len(dict_id2skeleton):
      dict_id2label = PoseRecognition.multiperson_classifier.classify(dict_id2skeleton)
      min_id = min(dict_id2skeleton.keys())
      # print("prediced label is :", dict_id2label[min_id])
      self.output = dict_id2label[min_id]

  def PostProcess(self):
    return self.output

# unit-test
if False:
  from pose_openpose_tf import PoseOpenpose

  import cv2
  image_reader = cv2.VideoCapture("/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/images/exercise.avi")

  pose = PoseOpenpose()
  pose.Setup()

  recog = PoseRecognition()
  recog.Setup()


  frame_id = 1
  while (frame_id < 32):
    print("Processing %dth image" % frame_id)

    # image = images_loader.read_image()
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