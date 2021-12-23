# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Given a list of images, run scene semantic segmentation using deeplab."""

import tensorflow as tf
from PIL import Image
import numpy as np

class Segmentation:
    def __init__(self, model_path):
        self.input_size = 513  # the model's input size, has to be this
        self.load_model(model_path) # load the deeplab model
        self.config_tensorflow()

    def load_model(self, model_path):
        graph = tf.Graph()
        with graph.as_default():
            gd = tf.GraphDef()
            with tf.gfile.GFile(model_path, "rb") as f:
                sg = f.read()
                gd.ParseFromString(sg)
                tf.import_graph_def(gd, name="")
            self.input_tensor = graph.get_tensor_by_name("ImageTensor:0")
            self.output_tensor = graph.get_tensor_by_name("SemanticPredictions:0")
        self.model = graph # save the model object

    def config_tensorflow(self):
        self.tfconfig = tf.ConfigProto()
        self.tfconfig.gpu_options.allow_growth = True
        self.tfconfig.gpu_options.visible_device_list = "%s" % (",".join(["%s" % i for i in [0]]))

    def run_model(self, imgs):
        seg_imgs = []
        with self.model.as_default():
            with tf.Session(graph=self.model, config=self.tfconfig) as sess:
                for img in imgs:
                    ori_img = Image.open(img)
                    # ori_img = Image.fromarray(img.astype(dtype=np.uint8))
                    w, h = ori_img.size
                    resize_r = 1.0 * self.input_size / max(w, h)
                    target_size = (int(resize_r * w), int(resize_r * h))
                    resize_img = ori_img.convert("RGB").resize(target_size, Image.ANTIALIAS)
                    seg_map = sess.run([self.output_tensor], feed_dict={self.input_tensor: [np.asarray(resize_img)]})
                    seg_map = seg_map[0][0]  # single image input test
                    seg_map = self.resize_seg_map(seg_map, 8.0)
                    seg_imgs.append(seg_map)
        return seg_imgs

    def resize_seg_map(self, seg, down_rate, keep_full=False):
        img_ = Image.fromarray(seg.astype(dtype=np.uint8))
        w_, h_ = img_.size
        neww, newh = int(w_ / down_rate), int(h_ / down_rate)
        if keep_full:
            neww, newh = 512, 288
        newimg = img_.resize((neww, newh))  # neareast neighbor
        newdata = np.array(newimg)
        return newdata

seg = Segmentation(model_path='deeplabv3_xception_ade20k_train/frozen_inference_graph.pb')
imgs = [f'0000_0_303_cam1/0000_0_303_cam1_F_000000{i}.jpg' for i in range(12,90,12)]
seg_imgs = seg.run_model(imgs)
print(seg_imgs)