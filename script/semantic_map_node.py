#!/usr/bin/env python3
import logging
import rospy
import roslib
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

sys.path.append(roslib.packages.get_pkg_dir("semantic_map") + "/include/omni3d/")

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import tf
import tf2_ros

from geometry_msgs.msg import TransformStamped

import cv2


class SemanticMap():
    def __init__(self, args) -> None:
        # ROS interface
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/marker_temp", Marker, queue_size=10)
        # self.img_sub = rospy.Subscriber("/image/test", Image, self.cb_segmentation_furniture, queue_size=10)
        self.img_sub = rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_raw", Image, self.cb_segmentation_furniture, queue_size=10)

        # tf配信用
        self.tf_br = tf2_ros.StaticTransformBroadcaster()
        self.tf_data = TransformStamped()

        # omni3dの結果をrosでpubish
        self.pub_seg_img = rospy.Publisher("/omni3d/result/segmentation", Image, queue_size=10)
        self.pub_novel_img = rospy.Publisher("/omni3d/result/novel_view", Image, queue_size=10)

        # omni3dの動作時に指定する引数
        # self.config_file = "cubercnn://omni3d/cubercnn_DLA34_FPN.yaml"
        # self.input_folder = "a"
        # self.threshold = 0.60
        # self.display = False
        # self.eval_only = True
        # self.num_gpus = 1
        # self.num_machines = 1
        # self.machine_rank = 0
        # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        # self.dist_url = "tcp://127.0.0.1:{}".format(port),
        # self.opts = None
        # self.opts = ['MODEL.WEIGHTS', self.config_file, 'OUTPUT_DIR', 'output/demo']


        # omni3Dの動作時に必要なモデルなどの設定
        self.args = args
        self.cfg = self.setup(self.args)
        self.model = build_model(self.cfg)
        
        logger.info("Model:\n{}".format(self.model))
        print(self.cfg.MODEL.WEIGHTS)
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=True
        )
        # omni3Dの動作時に必要なモデルなどの設定終了

        self.verts3D = None
        pass


    # def do_test(self, args, cfg, model):
    def cb_segmentation_furniture(self, img_msg):

        # list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')

        self.model.eval()

        thres = self.args.threshold

        output_dir = self.cfg.OUTPUT_DIR
        min_size = self.cfg.INPUT.MIN_SIZE_TEST
        max_size = self.cfg.INPUT.MAX_SIZE_TEST
        augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

        util.mkdir_if_missing(output_dir)

        category_path = os.path.join(util.file_parts(self.args.config_file)[0], 'category_meta.json')
            
        # store locally if needed
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

        metadata = util.load_json(category_path)
        cats = metadata['thing_classes']
        
        # for path in list_of_ims:

        # im_name = util.file_parts(path)[1]
        # im = cv2.imread(path)
        im = self.bridge.imgmsg_to_cv2(img_msg)
        # cv2.imshow("test", im)
        # cv2.waitKey(0)

        # if im is None:
        #     continue
        
        image_shape = im.shape[:2]  # h, w

        h, w = image_shape
        f_ndc = 4
        f = f_ndc * h / 2

        K = np.array([
            [f, 0.0, w/2], 
            [0.0, f, h/2], 
            [0.0, 0.0, 1.0]
        ])

        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
            'height': image_shape[0], 'width': image_shape[1], 'K': K
        }]

        dets = self.model(batched)[0]['instances']
        n_det = len(dets)

        meshes = []
        meshes_text = []

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                    dets.pred_pose, dets.scores, dets.pred_classes
                )):

                # skip
                if score < thres:
                    continue
                
                cat = cats[cat_idx]

                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)
        
        # print('File: {} with {} dets'.format(im_name, len(meshes)))
        print("detected ", len(meshes), "object")

        if len(meshes) > 0:
            im_drawn_rgb, im_topdown, _, verts3D = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
            print(verts3D)

            pub = rospy.Publisher("arrow_pub", Marker, queue_size=10)
            rate = rospy.Rate(25)

            # # publish
            # marker_data = Marker()
            # marker_data.header.frame_id = "map"
            # marker_data.header.stamp = rospy.Time.now()

            # marker_data.ns = "basic_shapes"
            # marker_data.id = 1

            # marker_data.action = Marker.ADD

            # # omni3dからの結果をもとにマーカーを出力する
            # # omni3dのxyzとrosでのxyzが一致していないことに注意
            # marker_data.pose.position.x = verts3D[0][0]
            # marker_data.pose.position.y = verts3D[0][2]
            # marker_data.pose.position.z = verts3D[0][1]

            # # marker_data.pose.position.x = 0
            # # marker_data.pose.position.y = 0
            # # marker_data.pose.position.z = 0


            # marker_data.pose.orientation.x=0.0
            # marker_data.pose.orientation.y=0.0
            # marker_data.pose.orientation.z=1.0
            # marker_data.pose.orientation.w=0.0

            # marker_data.color.r = 1.0
            # marker_data.color.g = 0.0
            # marker_data.color.b = 0.0
            # marker_data.color.a = 1.0

            # # 大きさを合わせて表示
            # scale_x = abs(verts3D[0][0] - verts3D[4][0])
            # scale_y = abs(verts3D[0][2] - verts3D[3][2])
            # scale_z = abs(verts3D[0][1] - verts3D[1][1])

            # marker_data.scale.x = scale_x
            # marker_data.scale.y = scale_y
            # marker_data.scale.z = scale_z

            # marker_data.lifetime = rospy.Duration(10)
            # marker_data.type = 1

            # pub.publish(marker_data)

            # tfとしてpublish
            # self.tf_data.header.stamp = rospy.Time.now()
            self.tf_data.header.frame_id = "head_rgbd_sensor_rgb_frame"
            self.tf_data.child_frame_id = "/detected/object"

            self.tf_data.transform.translation.x = verts3D[1][1]
            self.tf_data.transform.translation.y = verts3D[1][0]
            self.tf_data.transform.translation.z = verts3D[1][2]

            quat = tf.transformations.quaternion_from_euler(0, 0, 1)
            self.tf_data.transform.rotation.x = quat[0]
            self.tf_data.transform.rotation.y = quat[1]
            self.tf_data.transform.rotation.z = quat[2]
            self.tf_data.transform.rotation.w = quat[3]

            self.tf_br.sendTransform(self.tf_data)

            if self.args.display:
                im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
                vis.imshow(im_concat)

            # 結果をrosで出力

            # フォーマットを変更するための一時的対処
            path = roslib.packages.get_pkg_dir("semantic_map") + "/io/images/segmented_image.jpg"
            
            cv2.imwrite(path, im_drawn_rgb)
            im_segment = cv2.imread(path, 1)

            smi_segment = self.bridge.cv2_to_imgmsg(im_segment, encoding="bgr8")
            smi_novel_view = self.bridge.cv2_to_imgmsg(im_topdown, encoding="bgr8")

            self.pub_seg_img.publish(smi_segment)
            self.pub_novel_img.publish(smi_novel_view)

            # util.imwrite(im_drawn_rgb, os.path.join(output_dir, im_name+'_boxes.jpg'))
            # util.imwrite(im_topdown, os.path.join(output_dir, im_name+'_novel.jpg'))
        else:
            # util.imwrite(im, os.path.join(output_dir, im_name+'_boxes.jpg'))
            pass


    def setup(self, args):
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        get_cfg_defaults(cfg)

        config_file = args.config_file

        # store locally if needed
        if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
            config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

        cfg.merge_from_file(config_file)
        print(args.opts)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        default_setup(cfg, args)
        return cfg


    # def main(self, args):
    #     cfg = self.setup(args)
    #     model = build_model(cfg)
        
    #     logger.info("Model:\n{}".format(model))
    #     print(cfg.MODEL.WEIGHTS)
    #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=True
    #     )
    #     # print("a")
    #     self.do_test(args, cfg, model)


    #     return

class ArgsSetting():
    # omni3dの動作に必要な引数を設定するためのクラス
    # 動作はなし
    def __init__(self) -> None:
        self.config_file = "cubercnn://omni3d/cubercnn_DLA34_FPN.yaml"
        self.input_folder = "../include/omni3d/datasets/hma"
        self.threshold = 0.30
        self.display = False
        self.eval_only = True
        self.num_gpus = 1
        self.num_machines = 1
        self.machine_rank = 0
        port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        self.dist_url = "tcp://127.0.0.1:{}".format(port),
        self.opts = ['MODEL.WEIGHTS', "cubercnn://omni3d/cubercnn_DLA34_FPN.pth", 'OUTPUT_DIR', 'output/demo']
        print(self.opts)

if __name__ == "__main__":
    rospy.init_node("semantic_map")
    args = ArgsSetting()
    sm = SemanticMap(args)
    # print(args)

    rate = rospy.Rate(30)

    # launch(
    #     sm.main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )

    while not rospy.is_shutdown():
        # sm.pub_marker_test()
        # sm.main(args)
        rate.sleep()
        