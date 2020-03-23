
import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/Realtime-Action-Recognition/')
from utils.lib_openpose import SkeletonDetector
import utils.lib_images_io as lib_images_io
from utils.lib_classifier import ClassifierOnlineTest

from tracker import Tracker
# from multi_classifier import MultiPersonClassifier

# OPENPOSE_MODEL = "mobilenet_thin"
# OPENPOSE_IMG_SIZE = "656x368"

OPENPOSE_MODEL = "cmu"
OPENPOSE_IMG_SIZE = "432x368"

WINDOW_SIZE = 5

class MultiPersonClassifier(object):
  ''' This is a wrapper around ClassifierOnlineTest
      for recognizing actions of multiple people.
  '''

  def __init__(self, model_path, classes):

    self.dict_id2clf = {}  # human id -> classifier of this person

    # Define a function for creating classifier for new people.
    self._create_classifier = lambda human_id: ClassifierOnlineTest(
        model_path, classes, WINDOW_SIZE, human_id)

  def classify(self, dict_id2skeleton):
    ''' Classify the action type of each skeleton in dict_id2skeleton '''

    # Clear people not in view
    old_ids = set(self.dict_id2clf)
    cur_ids = set(dict_id2skeleton)
    humans_not_in_view = list(old_ids - cur_ids)
    for human in humans_not_in_view:
      del self.dict_id2clf[human]

    # Predict each person's action
    id2label = {}
    for id, skeleton in dict_id2skeleton.items():

      if id not in self.dict_id2clf:  # add this new person
        self.dict_id2clf[id] = self._create_classifier(id)

      classifier = self.dict_id2clf[id]
      id2label[id] = classifier.predict(skeleton)  # predict label
      # print("\n\nPredicting label for human{}".format(id))
      # print("  skeleton: {}".format(skeleton))
      # print("  label: {}".format(id2label[id]))

    return id2label

  def get_classifier(self, id):
    ''' Get the classifier based on the person id.
    Arguments:
        id {int or "min"}
    '''
    if len(self.dict_id2clf) == 0:
      return None
    if id == 'min':
      id = min(self.dict_id2clf.keys())
    return self.dict_id2clf[id]

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

  @staticmethod
  def Setup():
    PoseRecognition.skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    PoseRecognition.multiperson_tracker = Tracker()
    PoseRecognition.multiperson_classifier = MultiPersonClassifier("/home/yitao/Documents/fun-project/tensorflow-related/Realtime-Action-Recognition/model/trained_classifier.pickle", ['stand', 'walk', 'run', 'jump', 'sit', 'squat', 'kick', 'punch', 'wave'])

  def PreProcess(self, input):
    self.image = input

  def Apply(self):
    humans = PoseRecognition.skeleton_detector.detect(self.image)
    # print(humans)
    skeletons, scale_h = PoseRecognition.skeleton_detector.humans_to_skels_list(humans)
    # print(skeletons)
    # print(scale_h)
    skeletons = self.remove_skeletons_with_few_joints(skeletons)

    dict_id2skeleton = PoseRecognition.multiperson_tracker.track(skeletons)
    # print(dict_id2skeleton)

    if len(dict_id2skeleton):
      dict_id2label = PoseRecognition.multiperson_classifier.classify(dict_id2skeleton)
      # print(dict_id2label)
      min_id = min(dict_id2skeleton.keys())
      print("prediced label is :", dict_id2label[min_id])


  def PostProcess(self):
    pass

# unit-test
if True:
  images_loader = lib_images_io.ReadFromVideo("/home/yitao/Documents/fun-project/tensorflow-related/Realtime-Action-Recognition/data_test/exercise.avi", sample_interval = 1)

  recog = PoseRecognition()
  recog.Setup()


  frame_id = 1
  while (frame_id < 10):
    print("Processing %dth image" % frame_id)

    image = images_loader.read_image()
    recog.PreProcess(image)
    recog.Apply()
    recog.PostProcess()
    # print("[Yitao] len(skeletons) = %s" % len(skeletons))
    # print(skeletons[0])

    frame_id += 1
