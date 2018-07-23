import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# # Yitao-TLS-Begin
# import tensorflow as tf
# import os
# import sys
# from tensorflow.python.saved_model import builder as saved_model_builder
# from tensorflow.python.saved_model import signature_constants
# from tensorflow.python.saved_model import signature_def_utils
# from tensorflow.python.saved_model import tag_constants
# from tensorflow.python.saved_model import utils
# from tensorflow.python.util import compat

# tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
# FLAGS = tf.app.flags.FLAGS
# # Yitao-TLS-End

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # estimate human poses from a single image !
    image = common.read_imgfile(args.image, None, None)
    print("image shape = %s" % str(image.shape))
    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)
    t = time.time()
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    elapsed = time.time() - t

    logger.info('inference image: %s in %.4f seconds.' % (args.image, elapsed))

    print(humans)



    # # Yitao-TLS-Begin
    # export_path_base = "tf_openpose"
    # export_path = os.path.join(
    #     compat.as_bytes(export_path_base),
    #     compat.as_bytes(str(FLAGS.model_version)))
    # print('Exporting trained model to %s' % str(export_path))
    # builder = saved_model_builder.SavedModelBuilder(export_path)

    # tensor_info_x1 = tf.saved_model.utils.build_tensor_info(e.tensor_image)
    # tensor_info_x2 = tf.saved_model.utils.build_tensor_info(e.upsample_size)
    # tensor_info_y1 = tf.saved_model.utils.build_tensor_info(e.tensor_peaks)
    # tensor_info_y2 = tf.saved_model.utils.build_tensor_info(e.tensor_heatMat_up)
    # tensor_info_y3 = tf.saved_model.utils.build_tensor_info(e.tensor_pafMat_up)

    # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
    #     inputs={'tensor_image': tensor_info_x1,
    #                 'upsample_size': tensor_info_x2},
    #     outputs={'tensor_peaks': tensor_info_y1,
    #                 'tensor_heatMat_up': tensor_info_y2,
    #                 'tensor_pafMat_up': tensor_info_y3},
    #     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    # builder.add_meta_graph_and_variables(
    #     e.persistent_sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #       'predict_images':
    #         prediction_signature,
    #     },
    #     legacy_init_op=legacy_init_op)

    # builder.save()

    # print('Done exporting!')
    # # Yitao-TLS-End







    # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # a = fig.add_subplot(2, 2, 1)
    # a.set_title('Result')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # # show network output
    # a = fig.add_subplot(2, 2, 2)
    # plt.imshow(bgimg, alpha=0.5)
    # tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    # plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    # plt.colorbar()

    # tmp2 = e.pafMat.transpose((2, 0, 1))
    # tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    # tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    # a = fig.add_subplot(2, 2, 3)
    # a.set_title('Vectormap-x')
    # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    # plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    # plt.colorbar()

    # a = fig.add_subplot(2, 2, 4)
    # a.set_title('Vectormap-y')
    # # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    # plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    # plt.colorbar()
    # plt.show()
