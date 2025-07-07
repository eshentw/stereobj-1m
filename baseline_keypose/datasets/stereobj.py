import os
import json
import glob
import numpy as np
import tqdm
from PIL import Image

from .base import BaseDataset
from .box import Box


class StereoObjDataset(BaseDataset):
    def __init__(self, cfg, image_set):
        super().__init__(cfg, image_set)

    def _load_cam_params(self):
        # load camera parameters
        cam_param_filename = os.path.join(self.root_path, 'camera.json')
        with open(cam_param_filename, 'r') as f:
            cam_param = json.load(f)

        proj_matrix_l = cam_param['left']['P']
        proj_matrix_r = cam_param['right']['P']
        baseline = abs(proj_matrix_r[0][3] / proj_matrix_r[0][0])

        return proj_matrix_l, proj_matrix_r, baseline

    def _load_annotations(self, subdir, img_id):
        rt_path = os.path.join(
            self.root_path, subdir, img_id + '_rt_label.json')
        with open(rt_path, 'r') as f:
            rt_data = json.load(f)

        labels, boxes, kpts, R_origs = [], [], [], []

        for obj in rt_data['class']:
            obj_name = rt_data['class'][obj]

            if obj_name == "centrifuge_tube":
                bbox_filename = os.path.join(
                    self.root_path, 'objects', obj_name + '.bbox')
                with open(bbox_filename, 'r') as f:
                    bbox = f.read().split()
                    bbox = np.array([float(b) for b in bbox])
                    bbox = np.reshape(bbox, (3, 2)).T
                    x_max, x_min = bbox[:, 0]
                    y_max, y_min = bbox[:, 1]
                    z_max, z_min = bbox[:, 2]

                with open(os.path.join(
                        self.root_path, 'objects', obj_name + '.kp'), 'r') as f:
                    kps = f.read().split()
                    kps = np.array([float(k) for k in kps])
                    kps = np.reshape(kps, [-1, 3])

            else:
                kp_filename = os.path.join(
                    self.root_path, 'objects', obj_name + '.kp')
                with open(kp_filename, 'r') as f:
                    kps = f.read().split()
                    kps = np.array([float(k) for k in kps])
                    kps = np.reshape(kps, [-1, 3])

                    x_min, x_max = kps[:, 0].min(), kps[:, 0].max()
                    y_min, y_max = kps[:, 1].min(), kps[:, 1].max()
                    z_min, z_max = kps[:, 2].min(), kps[:, 2].max()

            length = x_max - x_min
            width = y_max - y_min
            height = z_max - z_min

            if obj_name in ['centrifuge_tube', 'screwdriver']:
                symmetric_type = "s"
            elif obj_name in ['microplate', 'tube_rack_1.5_2_ml',
                              'tube_rack_50_ml']:
                symmetric_type = "y"
            else:
                symmetric_type = "n"

            if obj_name in ['hammer', "screwdriver", "centrifuge_tube",
                            "wrench", "pipette_0.5_10",
                            "pipette_10_100", "pipette_100_1000"]:
                flip_pairs = [[0, 4], [3, 7], [1, 5], [2, 6]]
            elif obj_name in ['sterile_tip_rack_10',
                              'sterile_tip_rack_200', 'sterile_tip_rack_1000']:
                flip_pairs = [[0, 3], [4, 7], [5, 6], [1, 2]]
            elif obj_name in ["needle_nose_pliers", "side_cutters",
                              "wire_stripper"]:
                flip_pairs = [[0, 4], [3, 7], [1, 5], [2, 6],
                              [0, 3], [4, 7], [5, 6], [1, 2]]
            elif obj_name in ["microplate", "tube_rack_1.5_2_ml",
                              "tube_rack_50_ml"]:
                flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7],
                              [0, 3], [4, 7], [5, 6], [1, 2]]
            else:
                flip_pairs = []

            rt = rt_data['rt'][obj]
            R = np.array(rt['R'])
            t = np.array(rt['t'])

            # Ensure length is the longest, followed by width, then height
            dimensions = np.abs(np.array([length, width, height]))
            sorted_indices = np.argsort(dimensions)[::-1]
            length, width, height = dimensions[sorted_indices]
            R_orig = R.copy()
            R = R[:, sorted_indices]  # Reorder the rotation matrix accordingly

            # Re-orthonormalize R to ensure it remains a valid rotation matrix
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                R[:, -1] *= -1
            # Check if the rotation matrix is valid
            if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
                raise ValueError("Rotation matrix is not orthogonal")

            size = [length, width, height]

            box = Box(size, R, t)
            box.update_flip_pairs(flip_pairs)
            box.update_symmetric_type(symmetric_type)

            found_key = None
            for key in self.label_class_mapping:
                if key in obj_name:
                    found_key = key
                    break

            if not found_key:
                raise KeyError(f"Object name '{obj_name}' "
                               "does not match any key in label_class_mapping")

            labels.append(self.label_class_mapping[found_key])
            boxes.append(box)
            kpts.append(kps[:self.n_kpts])
            R_origs.append(R_orig)

        return {'labels': labels, 'boxes': boxes, "nb_kpts_3d": kpts, "R_origs": R_origs}

    def _load_image(self, img_path):
        img = Image.open(img_path)
        width, height = img.size
        img_l = img.crop((0, 0, width // 2, height))
        img_r = img.crop((width // 2, 0, width, height))
        return img_l, img_r

    def _get_db(self):
        gt_db = []

        if self.target_obj is None:
            if self.image_set == "train":
                split_filenames = glob.glob(os.path.join(
                    self.root_path, 'split', 'train_*.json'))
            elif self.image_set == "val":
                split_filenames = glob.glob(os.path.join(
                    self.root_path, 'split', 'val_*.json'))
            else:
                raise ValueError("Invalid image set")
        else:
            if self.image_set == "train":
                split_filenames = glob.glob(os.path.join(
                    self.root_path, 'split', f'train_{self.target_obj}.json'))
            elif self.image_set == "val":
                split_filenames = glob.glob(os.path.join(
                    self.root_path, 'split', f'val_{self.target_obj}.json'))
            else:
                raise ValueError("Invalid image set")

        # load image info: dirname and img_id
        filename_dict = {}
        gt_db = []
        for split_filename in sorted(split_filenames):
            with open(split_filename, 'r') as f:
                filename_dict.update(json.load(f))
        bad_scenes = [
            "mechanics_scene_7_08162020_5",
            "biolab_scene_2_07312020_7",
            "biolab_scene_2_07312020_11",
            "biolab_scene_2_07312020_13",
        ]
        for scene in bad_scenes:
            filename_dict.pop(scene, None)

        # Load camera parameters
        proj_matrix_l, proj_matrix_r, baseline = self._load_cam_params()

        for subdir in tqdm.tqdm(filename_dict):
            for img_id in filename_dict[subdir]:
                # Skip every 3 frames
                # if int(img_id.split('_')[-1]) % 3 != 0:
                #     continue
                if int(img_id.split('_')[-1]) % 300 != 0:
                    continue
                annots = self._load_annotations(subdir, img_id)
                annots.update({
                    'img_id': img_id,
                    'subdir': subdir,
                    'img_path': os.path.join(
                        self.root_path, subdir, img_id + '.jpg'),
                    'proj_matrix_l': proj_matrix_l,
                    'proj_matrix_r': proj_matrix_r,
                    'baseline': baseline,
                })

                gt_db.append(annots)

        return gt_db
