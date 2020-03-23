import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')
from utils.lib_classifier import ClassifierOnlineTest

class MultiPersonClassifier(object):
  ''' This is a wrapper around ClassifierOnlineTest
      for recognizing actions of multiple people.
  '''

  def __init__(self, model_path, classes):
    WINDOW_SIZE = 5

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