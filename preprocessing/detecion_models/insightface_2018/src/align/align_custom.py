from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import tensorflow as tf
import numpy as np
try:
    from . import detect_face
except:
    import detect_face

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_preprocess
import cv2

def squarify(M,val=0):
    (a, b, c)=M.shape
    if a==b:
        return M
    if a>b:
        diff = a-b
        left_diff = diff // 2
        right_diff = diff - left_diff
        padding=((0,0), (left_diff,right_diff), (0,0))
    else:
        diff = b-a
        top_diff = diff // 2
        btm_diff = diff - top_diff
        padding=((top_diff, btm_diff), (0,0), (0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe, GTframe):
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    return ratio


class Aligner():

    def __init__(self):

        with tf.Graph().as_default():
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
            #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            sess = tf.Session(config=config)
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.9]
        self.factor = 0.85
        self.image_size = "112,112"

    def img_read(self, image_path):
        img = misc.imread(image_path)
        return img

    def align_img(self, img, return_fail_as_None=False):

        if img.ndim == 2:
            img = to_rgb(img)
        img = img[:, :, 0:3]

        _bbox = None
        _landmark = None
        bounding_boxes, points = detect_face.detect_face(img, 
                                                         self.minsize, 
                                                         self.pnet, 
                                                         self.rnet, 
                                                         self.onet,
                                                         self.threshold, 
                                                         self.factor)

        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            img_size = np.asarray(img.shape)[0:2]
            bindex = 0
            if nrof_faces > 1:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                        (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                bindex = np.argmax(bounding_box_size -
                                    offset_dist_squared * 2.0)  # some extra weight on the centering
            _bbox = bounding_boxes[bindex, 0:4]
            _landmark = points[:, bindex].reshape((2, 5)).T
        
        else:
            if return_fail_as_None:
                return None

        warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size=self.image_size)
        bgr = warped[..., ::-1]
        return bgr

    def infer_align_batch(self, images, return_image=False, failed_image_behavior='center_crop'):
        assert failed_image_behavior in ['center_crop', 'identity', 'pad_high']

        # i used it to bad images for ijbs after using single to align all
        res = detect_face.bulk_detect_face(images,
                                     minsize=self.minsize,
                                     pnet=self.pnet,
                                     rnet=self.rnet,
                                     onet=self.onet,
                                     threshold=self.threshold,
                                     factor=self.factor)
        # res: list where fail would contain None

        if return_image:
            bgr_images = []
            assert len(images) == len(res)
            for idx, pred_info in enumerate(res):
                img = images[idx]

                if pred_info is None:
                    # this would result in center cropped img
                    _bbox = None
                    _landmark = None
                else:
                    bounding_boxes, points = pred_info
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(img.shape)[0:2]
                        bindex = 0
                        if nrof_faces > 1:
                            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                            img_center = img_size / 2
                            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                            bindex = np.argmax(bounding_box_size -
                                                offset_dist_squared * 2.0)  # some extra weight on the centering
                        _bbox = bounding_boxes[bindex, 0:4]
                        _landmark = points[:, bindex].reshape((2, 5)).T
                    else:
                        # this would result in center cropped img
                        _bbox = None
                        _landmark = None

                if failed_image_behavior == 'identity' and _bbox is None:
                    bgr = img[..., ::-1]
                    tgt_img_size = [int(k) for k in self.image_size.split(',')]
                    bgr = cv2.resize(bgr, tgt_img_size)
                elif failed_image_behavior == 'pad_high' and _bbox is None:
                    bgr = img[..., ::-1]
                    pad_ratio = 0.3
                    pad_side = int(pad_ratio * 32)
                    high_padding=((pad_side, 0), (0,0), (0,0))
                    bgr = np.pad(bgr, high_padding,mode='constant',constant_values=0)
                    bgr = squarify(bgr)
                    tgt_img_size = [int(k) for k in self.image_size.split(',')]
                    bgr = cv2.resize(bgr, tgt_img_size)
                else:
                    warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size=self.image_size)
                    bgr = warped[..., ::-1]
                bgr_images.append(bgr)
            return res, bgr_images
        else:
            return res