import copy
import torch

from .transforms import make_transforms
from .utils import (
    to_tensor, kpts_to_boxes, box_xyxy_to_cxcywh, kpts_pose_disp_to_left_right
)


class BaseDataset:
    def __init__(self, cfg, image_set):
        self.root_path = cfg.root
        self.image_set = image_set
        self.n_kpts = cfg.n_kpts

        class_names = dict(cfg.names)
        self.label_class_mapping = {v: k for k, v in class_names.items()}
        self.target_obj = cfg.get('target_obj', None)

        self.db = self._get_db()
        # does not perform horizontal flip
        self.transform = make_transforms(cfg, image_set)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        # validate that all required keys are present in db_rec
        required_keys = ["img_path", "labels", "boxes",
                         "proj_matrix_l", "proj_matrix_r", "baseline"]
        missing = [k for k in required_keys if k not in db_rec]
        if missing:
            raise KeyError(f"db_rec is missing required keys: {missing}")

        img_l, img_r = self._load_image(db_rec['img_path'])
        image = {"left": img_l, "right": img_r}

        nb_kpts_3d = db_rec.get("nb_kpts_3d", None)
        target = {
            "labels": to_tensor(db_rec['labels'], dtype=torch.int64),
            "boxes": db_rec['boxes'],
            "proj_matrix_l": to_tensor(
                db_rec['proj_matrix_l'], dtype=torch.float64),
            "proj_matrix_r": to_tensor(
                db_rec['proj_matrix_r'], dtype=torch.float64),
            "baseline": to_tensor(
                [db_rec['baseline']], dtype=torch.float64),
            "nb_kpts_3d": to_tensor(nb_kpts_3d, dtype=torch.float64) if nb_kpts_3d is not None else None,
            "R_origs": to_tensor(db_rec.get('R_origs', []), dtype=torch.float64)
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        for key, value in target.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Target value for {key} is not a tensor")

        kpts_l, kpts_r = kpts_pose_disp_to_left_right(
            target["kpts_pose"], target["kpts_disp"])
        target["boxes_l"] = box_xyxy_to_cxcywh(kpts_to_boxes(kpts_l))
        target["boxes_r"] = box_xyxy_to_cxcywh(kpts_to_boxes(kpts_r))

        # assign any remaining keys from db_rec into target
        for key, value in db_rec.items():
            if key not in required_keys and key not in target:
                target[key] = value

        return image, target

    def _get_db(self):
        raise NotImplementedError("Subclasses must implement _get_db")

    def _load_image(self, img_path):
        raise NotImplementedError(
            "_load_image method must be implemented by subclasses"
        )
