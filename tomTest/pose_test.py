import time

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

if __name__ == '__main__':

    image_path = '/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/images/p1.jpg'
    resize_out_ratio = 4.0
    model_name = 'cmu'

    e = TfPoseEstimator(get_graph_path(model_name), target_size=(432, 368))
    
    iteration_list = [10]
    for iteration in iteration_list:
        start = time.time()
        for i in range(iteration):
            # estimate human poses from a single image !
            # image = common.read_imgfile(image_path, None, None)
            image = cv2.imread(image_path)
            print("image shape = %s" % str(image.shape))
            if image is None:
                sys.exit(-1)
            t = time.time()
            humans = e.inference(image, resize_to_default=False, upsample_size=resize_out_ratio)
            elapsed = time.time() - t

        end = time.time()
        print("It takes %s sec to run %d images for tf-openpose" % (str(end - start), iteration))



    print(image.shape)
    print(humans)

    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    import matplotlib.pyplot as plt

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
