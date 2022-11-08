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

import cv2


class SemanticMap():
    def __init__(self) -> None:
        self.pub = rospy.Publisher("/marker_temp", Marker, queue_size=10)
        self.img_sub = rospy.Subscriber("/image/temp", Image, queue_size=10)

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

        self.verts3D = None
        pass

    def pub_marker_test(self) -> None:
        # publish
        marker_data = Marker()
        marker_data.header.frame_id = "map"
        marker_data.header.stamp = rospy.Time.now()

        marker_data.ns = "basic_shapes"
        marker_data.id = 1

        marker_data.action = Marker.ADD

        # marker_data.pose.position.x = verts3D[0][0]
        # marker_data.pose.position.y = verts3D[0][1]
        # marker_data.pose.position.z = verts3D[0][2]

        # -0.29077148  0.34356543  2.5544643
        marker_data.pose.position.x = -0.29077148
        marker_data.pose.position.y = 0.34356543
        marker_data.pose.position.z = 2.5544643


        marker_data.pose.orientation.x=0.0
        marker_data.pose.orientation.y=0.0
        marker_data.pose.orientation.z=1.0
        marker_data.pose.orientation.w=0.0

        marker_data.color.r = 1.0
        marker_data.color.g = 0.0
        marker_data.color.b = 0.0
        marker_data.color.a = 1.0

        marker_data.scale.x = 1
        marker_data.scale.y = 1.1
        marker_data.scale.z = 1.1

        marker_data.lifetime = rospy.Duration(10)
        marker_data.type = 0

        self.pub.publish(marker_data)


    def do_test(self, args, cfg, model):

        list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')

        model.eval()

        thres = args.threshold

        output_dir = cfg.OUTPUT_DIR
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

        util.mkdir_if_missing(output_dir)

        category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')
            
        # store locally if needed
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

        metadata = util.load_json(category_path)
        cats = metadata['thing_classes']
        
        for path in list_of_ims:

            im_name = util.file_parts(path)[1]
            im = cv2.imread(path)

            if im is None:
                continue
            
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

            dets = model(batched)[0]['instances']
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
            
            print('File: {} with {} dets'.format(im_name, len(meshes)))

            if len(meshes) > 0:
                im_drawn_rgb, im_topdown, _, verts3D = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
                print(verts3D)

                pub = rospy.Publisher("arrow_pub", Marker, queue_size=10)
                rate = rospy.Rate(25)

                # publish
                marker_data = Marker()
                marker_data.header.frame_id = "map"
                marker_data.header.stamp = rospy.Time.now()

                marker_data.ns = "basic_shapes"
                marker_data.id = 1

                marker_data.action = Marker.ADD

                marker_data.pose.position.x = verts3D[0][0]
                marker_data.pose.position.y = verts3D[0][1]
                marker_data.pose.position.z = verts3D[0][2]

                # marker_data.pose.position.x = 0
                # marker_data.pose.position.y = 0
                marker_data.pose.position.z = 0


                marker_data.pose.orientation.x=0.0
                marker_data.pose.orientation.y=0.0
                marker_data.pose.orientation.z=1.0
                marker_data.pose.orientation.w=0.0

                marker_data.color.r = 1.0
                marker_data.color.g = 0.0
                marker_data.color.b = 0.0
                marker_data.color.a = 1.0

                marker_data.scale.x = 1
                marker_data.scale.y = 1.1
                marker_data.scale.z = 1.1

                marker_data.lifetime = rospy.Duration(10)
                marker_data.type = 1

                pub.publish(marker_data)

                if args.display:
                    im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
                    vis.imshow(im_concat)

                util.imwrite(im_drawn_rgb, os.path.join(output_dir, im_name+'_boxes.jpg'))
                util.imwrite(im_topdown, os.path.join(output_dir, im_name+'_novel.jpg'))
            else:
                util.imwrite(im, os.path.join(output_dir, im_name+'_boxes.jpg'))


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


    def main(self, args):
        cfg = self.setup(args)
        model = build_model(cfg)
        
        logger.info("Model:\n{}".format(model))
        print(cfg.MODEL.WEIGHTS)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        # print("a")
        self.do_test(args, cfg, model)


        return

class ArgsSetting():
    # omni3dの動作に必要な引数を設定するためのクラス
    # 動作はなし
    def __init__(self) -> None:
        self.config_file = "cubercnn://omni3d/cubercnn_DLA34_FPN.yaml"
        self.input_folder = "../include/omni3d/datasets/hma"
        self.threshold = 0.60
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
    sm = SemanticMap()
    args = ArgsSetting()
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
        sm.main(args)
        rate.sleep()
        