import time

import sys
sys.path.append('/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/')
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# Yitao-TLS-Begin
import tensorflow as tf
import os
import sys
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS
# Yitao-TLS-End

if __name__ == '__main__':

    image_path = '/home/yitao/Documents/fun-project/tensorflow-related/tf-pose-estimation/images/p1.jpg'
    resize_out_ratio = 4.0
    # model_name = 'cmu'
    model_name = "mobilenet_thin"

    e = TfPoseEstimator(get_graph_path(model_name), target_size=(432, 368))
    
    iteration_list = [10]
    for iteration in iteration_list:
        for i in range(iteration):
            start = time.time()
            # estimate human poses from a single image !
            image = common.read_imgfile(image_path, None, None)
            # print("image shape = %s" % str(image.shape))
            if image is None:
                sys.exit(-1)
            t = time.time()
            humans = e.inference(image, resize_to_default=False, upsample_size=resize_out_ratio)
            elapsed = time.time() - t

            end = time.time()
            print("It takes %s sec to run" % (str(end - start)))

    # Yitao-TLS-Begin
    if (model_name == "cmu"):
        export_path_base = "pose_openpose"
    else:
        export_path_base = "pose_thinpose"
    export_path = os.path.join(
        compat.as_bytes(export_path_base),
        compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to %s' % str(export_path))
    builder = saved_model_builder.SavedModelBuilder(export_path)

    tensor_info_x1 = tf.saved_model.utils.build_tensor_info(e.tensor_image)
    tensor_info_x2 = tf.saved_model.utils.build_tensor_info(e.upsample_size)
    # tensor_info_y = tf.saved_model.utils.build_tensor_info(e.tensor_output)
    tensor_info_y1 = tf.saved_model.utils.build_tensor_info(e.tensor_peaks)
    tensor_info_y2 = tf.saved_model.utils.build_tensor_info(e.tensor_heatMat_up)
    tensor_info_y3 = tf.saved_model.utils.build_tensor_info(e.tensor_pafMat_up)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'tensor_image': tensor_info_x1,
                    'upsample_size': tensor_info_x2},
        # outputs = {'tensor_output': tensor_info_y},
        outputs={'tensor_peaks': tensor_info_y1,
                    'tensor_heatMat_up': tensor_info_y2,
                    'tensor_pafMat_up': tensor_info_y3},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        e.persistent_sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
          'predict_images':
            prediction_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')
    # Yitao-TLS-End

    print(image.shape)
    print(humans)
