import os
import time
import json
import copy
from functools import partial

import numpy as np

import open3d
from open3d.pipelines.registration import TukeyLoss
from open3d.pipelines.registration import registration_icp
from open3d.pipelines.registration import ICPConvergenceCriteria
from open3d.pipelines.registration import TransformationEstimationPointToPlane
from open3d.visualization import draw_geometries_with_key_callbacks as visualization_base
from typing import Dict, List

step = .01
key_mapping_when_translating = {
    ord('A'): np.array([[1, 0, 0, -step], [0., 1., 0., 0.], [0., 0., 1., 0.], [0, 0, 0, 1]]),
    ord("D"): np.array([[1., 0, 0, step], [0., 1., 0., 0.], [0., 0., 1., 0.], [0, 0, 0, 1]]),
    ord('S'): np.array([[1., 0., 0., 0.], [0, 1, 0, -step], [0., 0., 1., 0.], [0, 0, 0, 1]]),
    ord('W'): np.array([[1., 0., 0., 0.], [0., 1, 0, step], [0., 0., 1., 0.], [0, 0, 0, 1]]),
    ord("E"): np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0, 0, 1, -step], [0, 0, 0, 1]]),
    ord('Q'): np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0, 0, 1., step], [0, 0, 0, 1]])
}
c, s = np.cos(np.pi / 36), np.sin(np.pi / 36)
key_mappings_when_rotating = {
    ord('W'): np.array([[c, -s, 0, 0], [s, c, 0., 0], [0, 0, 1., 0.], [0, 0, 0, 1]]),
    ord('S'): np.array([[c, s, 0., 0], [-s, c, 0, 0], [0, 0, 1., 0.], [0, 0, 0, 1]]),
    ord('A'): np.array([[c, 0., s, 0], [0, 1, 0., 0], [-s, 0., c, 0], [0, 0, 0, 1]]),
    ord("D"): np.array([[c, 0, -s, 0], [0, 1., 0, 0], [s, 0., c, 0.], [0, 0, 0, 1]]),
    ord('Q'): np.array([[1., 0, 0, 0], [0, c, -s, 0], [0., s, c, 0.], [0, 0, 0, 1]]),
    ord("E"): np.array([[1., 0, 0, 0], [0, c, s, 0.], [0, -s, c, 0.], [0, 0, 0, 1]]),
}


class ManualRegistrationTool:
    def __init__(self, source, target, init=np.eye(4, dtype=np.float64), add_local_frame=True, **window_kwargs):
        self.source = source
        self.target = target

        self.transformation = init
        self.is_translating = True

        self.source_visible = True
        self.target_visible = True
        self.local_frame_visible = add_local_frame

        self.previous_key = -1
        self.previous_key_time = time.time()
        self.down_time = 0.0
        self.speed_up_scale = 1.0

        visualization_base(geometry_list=self.compose(),
                           key_to_callback={
                               ord(c): partial(self.dispatch, key=ord(c))
                               for c in ("W", "S", "A", "D", "Q", "E", "M", "T", "I", "H")
                           },
                           **window_kwargs)

    def compose(self) -> List[open3d.geometry.Geometry]:
        out = []
        with open("settings.json", "r") as file:
            settings = json.load(file).get("composition", {})
        if self.source_visible:
            source = copy.deepcopy(self.source).transform(self.transformation)
            voxel_size = settings.get("source", {}).get("voxel_size", None)
            color = settings.get("source", {}).get("color", None)
            if voxel_size is not None:
                source = source.voxel_down_sample(voxel_size)
            if color is not None:
                source.paint_uniform_color(np.asarray(color))
            out.append(source)
        if self.target_visible:
            target = copy.deepcopy(self.target)
            voxel_size = settings.get("target", {}).get("voxel_size", None)
            color = settings.get("target", {}).get("color", None)
            if voxel_size is not None:
                target = target.voxel_down_sample(voxel_size)
            if color is not None:
                target.paint_uniform_color(np.asarray(color))
            out.append(target)
        if self.local_frame_visible:
            out.append(open3d.geometry.TriangleMesh().create_coordinate_frame().transform(self.transformation))
        return out

    def refine_with_icp(self):
        with open("settings.json", "r") as file:
            settings = json.load(file).get("icp", {})
        factor = settings.get("factor", 1.0)
        scale_transform = np.diag([factor, factor, factor, 1.0])

        source = copy.deepcopy(self.source).transform(self.transformation).transform(scale_transform)
        target = copy.deepcopy(self.target).transform(scale_transform)

        voxel_size = settings.get("voxel_size", None)
        if voxel_size is not None:
            source = source.voxel_down_sample(voxel_size)
            target = target.voxel_down_sample(voxel_size)

        normals_estimation_search_params = open3d.geometry.KDTreeSearchParamHybrid(radius=.04, max_nn=30)
        source.estimate_normals(normals_estimation_search_params)
        target.estimate_normals(normals_estimation_search_params)

        open3d_registration_icp_kwargs = {
            "max_correspondence_distance": settings.get("max_correspondence_distance", .3),
            "estimation_method": TransformationEstimationPointToPlane(TukeyLoss(k=.1)),
            "criteria": ICPConvergenceCriteria(relative_fitness=1E-6, relative_rmse=1E-6, max_iteration=25)
        }
        result = registration_icp(source=source,
                                  target=target, **open3d_registration_icp_kwargs)
        matrix = result.transformation
        matrix = scale_transform @ matrix @ np.linalg.inv(scale_transform)
        self.transformation = matrix @ self.transformation

    def dispatch(self, visualizer: open3d.visualization.Visualizer, key):
        if key == ord("H"):
            print("HELP")
        elif key == ord("M"):
            self.is_translating = not self.is_translating
            print("TRANSLATING" if self.is_translating else "ROTATING")
            return True
        elif key == ord("I"):
            self.refine_with_icp()
        elif key == ord("T"):
            if self.source_visible and self.target_visible:
                self.source_visible = False
            elif self.target_visible:
                self.source_visible = True
                self.target_visible = False
            else:
                self.source_visible = True
                self.target_visible = True
        else:
            if self.previous_key == key:
                time_since_last_event = time.time() - self.previous_key_time
                if time_since_last_event < 0.05:
                    self.down_time += time.time() - self.previous_key_time
                else:
                    self.down_time = 0
                self.speed_up_scale = 1 + int(self.down_time / 2)
                self.previous_key_time = time.time()
            self.previous_key = key

            matrix = (key_mapping_when_translating if self.is_translating else key_mappings_when_rotating)[key]
            for i in range(int(self.speed_up_scale)):
                self.transformation = self.transformation @ matrix

        visualizer.clear_geometries()
        for element in self.compose():
            visualizer.add_geometry(element, reset_bounding_box=False)
        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--source", type=str, default="a.ply", help="source point cloud.")
    parser.add_argument('-t', "--target", type=str, default="b.ply", help="target point cloud.")
    parser.add_argument('-o', "--output", type=str, default="output.txt", help="save path.")
    parser.add_argument('-i', "--init", type=str, default=None, help="initial guess.")
    parser.add_argument('-v', "--verbose", action="store_true", help="add print messages.")
    args = parser.parse_args()

    point_time = time.time()
    transformation = np.eye(4, dtype=np.float64)
    if args.init is not None and os.path.isfile(args.init):
        transformation = np.loadtxt(args.init)
    tool = ManualRegistrationTool(source=open3d.io.read_point_cloud(args.source),
                                  target=open3d.io.read_point_cloud(args.target),
                                  init=transformation)
    np.savetxt(args.output, tool.transformation)
    duration = time.time() - point_time
