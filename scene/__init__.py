#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, SceneInfo
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, prog_train_interval=200, dataset_size=1000, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.prog_train_interval = prog_train_interval
        self.dataset_size = dataset_size
        self.resolution_scales = resolution_scales
        self.args = args
        self.scene_info = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            self.scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            self.scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # select images to form a sub-set for training
        self.downsample_train_set(self.scene_info, self.dataset_size)

        if not self.loaded_iter:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            json_cams = [camera_to_JSON(id, cam) for id, cam in enumerate(camlist)]
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, args)
            print("Test camera: ", len(self.test_cameras[resolution_scale]))

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    f"iteration_{str(self.loaded_iter)}",
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def res_scale(self, no_prog_subset):
        start = no_prog_subset * self.prog_train_interval
        end = (no_prog_subset + 1) * self.prog_train_interval
        end = min(end, len(self.scene_info.train_cameras))
        for resolution_scale in self.resolution_scales:
            print(f"Loading Training Cameras from {start} to {end}")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras[start:end], resolution_scale, self.args)
            print("Train camera: ", len(self.train_cameras[resolution_scale]))

    def shuffle(self):
        random.shuffle(self.scene_info.train_cameras)

    def getTrainCameras(self, scale=1.0):        
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    

    def downsample_train_set(self, scene_info, dataset_size):
        if dataset_size < len(scene_info.train_cameras):
            length = len(scene_info.train_cameras)
            interval = length / dataset_size
            index = 0
            selected_train_cameras = []
            print(scene_info.train_cameras)
            while length < dataset_size and index < length:
                selected_train_cameras.append(scene_info.train_cameras[int(index)])
                index += interval
            # create a new class to replace the old one
            self.scene_info = SceneInfo(
                point_cloud=scene_info.point_cloud,
                train_cameras=selected_train_cameras,
                test_cameras=scene_info.test_cameras,
                nerf_normalization=scene_info.nerf_normalization,
                ply_path=scene_info.ply_path
            )
            print(f"Using subset of {len(self.scene_info.train_cameras)}/{dataset_size} cameras")
        else:
            print(f"Using full training set of {len(scene_info.train_cameras)} cameras")