import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path

import cv2

class PoseOpenpose:
  @staticmethod
  def Setup():
    PoseOpenpose.e = TfPoseEstimator(get_graph_path("cmu"), target_size=(432, 368))
    PoseOpenpose.resize_out_ratio = 4.0

  def PreProcess(self, input):
    self.image = input

  def Apply(self):
    self.humans = PoseOpenpose.e.inference(self.image, resize_to_default=False, upsample_size=PoseOpenpose.resize_out_ratio)

  def PostProcess(self):
    return self.humans

# unit-test
if False:
  pose = PoseOpenpose()
  pose.Setup()

  image_path = '/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/images/p1.jpg'
  image = cv2.imread(image_path)

  pose.PreProcess(image)
  pose.Apply()
  humans = pose.PostProcess()

  print(humans)