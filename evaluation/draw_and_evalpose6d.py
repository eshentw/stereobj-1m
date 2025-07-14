import argparse
import cv2
import numpy as np
import json
import os
import sys
import torch
from collections import Counter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baseline_keypose.pose6d_validator import Pose6DValidator
from baseline_keypose.datasets.utils import project_3d_to_2d_batch
from baseline_keypose.datasets.box import get_canonical_box
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# import evaluate_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='', help='GPU to use [default: ]')
parser.add_argument('--input_json', default='', help='Input json filename [default: ]')
parser.add_argument('--out_dir', default='', help='Output directory [default: ]')
parser.add_argument('--refactor_gt_json', default='', help='GT json filename [default: ]')
parser.add_argument('--root_dir', default='', help='Root directory for data [default: ]')
parser.add_argument('--store_image', action='store_true', help='Whether to store the results [default: False]')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def draw_box(box_l, box_r, img_l, img_r, color):
    if len(box_l) >= 8:
        box_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
        ]
        
        # Draw edges for left image
        for edge in box_edges:
            pt1_l = (int(box_l[edge[0], 0]), int(box_l[edge[0], 1]))
            pt2_l = (int(box_l[edge[1], 0]), int(box_l[edge[1], 1]))
            cv2.line(img_l, pt1_l, pt2_l, color, 4)
        
        # Draw edges for right image
        for edge in box_edges:
            pt1_r = (int(box_r[edge[0], 0]), int(box_r[edge[0], 1]))
            pt2_r = (int(box_r[edge[1], 0]), int(box_r[edge[1], 1]))
            cv2.line(img_r, pt1_r, pt2_r, color, 4)
    else:
        raise ValueError("Box must have at least 8 points to draw edges.")

if __name__ == "__main__":
    classes = [
        "blade_razor",
        "hammer",
        "needle_nose_pliers",
        "screwdriver",
        "side_cutters",
        "tape_measure",
        "wire_stripper",
        "wrench",
        "centrifuge_tube",
        "microplate",
        "tube_rack",
        "pipette",
        "sterile_tip_rack"
    ]
    class_to_label = {
        'blade_razor': 0,
        'hammer': 1,
        'needle_nose_pliers': 2,
        'screwdriver': 3,
        'side_cutters': 4,
        'tape_measure': 5,
        'wire_stripper': 6,
        'wrench': 7,
        'centrifuge_tube': 8,
        'microplate': 9,
        'tube_rack': 10,
        'pipette': 11,
        'sterile_tip_rack': 12
    }
    label_to_class = {
        0: 'blade_razor',
        1: 'hammer',
        2: 'needle_nose_pliers',
        3: 'screwdriver',
        4: 'side_cutters',
        5: 'tape_measure',
        6: 'wire_stripper',
        7: 'wrench',
        8: 'centrifuge_tube',
        9: 'microplate',
        10: 'tube_rack',
        11: 'pipette',
        12: 'sterile_tip_rack'
    }
    colors_rgb = [
        (0, 255, 255),
        (162, 255, 0),
        (255, 230, 0),
        (170, 0, 255),
        (0, 255, 170),
        (85, 0, 255),
        (0, 255, 85),
        (255, 0, 0),
        (255, 0, 170),
        (49, 255, 0),
        (0, 170, 255),
        (255, 112, 0),
        (0, 85, 255),
    ]

    pose6d_validator = Pose6DValidator(classes, use_matches_for_pose=True)
    with open(args.input_json, 'r') as f:
        input_dict = json.load(f)
    pred_dict = input_dict['pred']

    with open(args.refactor_gt_json, 'r') as f:
        gt_dict = json.load(f)

    assert input_dict['split'] == gt_dict['split']
    
    del gt_dict['split']

    for seq_id in tqdm(gt_dict):
        frame_ids = list(gt_dict[seq_id].keys())
        for i in range(0, len(frame_ids), 10):
            frame_id = frame_ids[i]
            draw_meta_info = {
                "seq_id": seq_id,
                "frame_id": frame_id,
                "cls_type": [],
                "gt_pose": [],
                "pred_size": [],
                "pred_pose": []
            }
            for cls_type in gt_dict[seq_id][frame_id]:
                try:
                    assert seq_id in pred_dict[cls_type]
                    assert frame_id in pred_dict[cls_type][seq_id]
                except:
                    continue
                target_label_id = class_to_label.get(cls_type, None)

                if cls_type.startswith('sterile_tip_rack'):
                    target_label_id = class_to_label.get('sterile_tip_rack', None)
                elif cls_type.startswith('tube_rack'):
                    target_label_id = class_to_label.get('tube_rack', None)
                elif cls_type.startswith('pipette'):
                    target_label_id = class_to_label.get('pipette', None)
                assert target_label_id is not None, f"Class {cls_type} not found in dataset classes."

                pose_pred = pred_dict[cls_type][seq_id][frame_id]["pose_pred"]
                pose_pred = torch.tensor(pose_pred, dtype=torch.float32)
                if pose_pred.ndim == 2:
                    pose_pred = pose_pred.unsqueeze(0)

                pose_gt = gt_dict[seq_id][frame_id][cls_type]
                pose_gt = torch.tensor(pose_gt, dtype=torch.float32)
                if pose_gt.ndim == 2:
                    pose_gt = pose_gt.unsqueeze(0)

                pred_pose = [torch.cat((pose, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)), dim=0).numpy()
                             for pose in pose_pred]
                gt_pose = [torch.cat((pose, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)), dim=0).numpy()
                           for pose in pose_gt]

                # dummy size
                pred_size = [pred_dict[cls_type][seq_id][frame_id]["size"]]
                gt_size = pred_size.copy()

                # dummy labels and scores
                t_label = torch.tensor([target_label_id], dtype=torch.int64)
                labels = t_label.repeat(len(pred_pose))
                scores = torch.ones(len(pred_pose), dtype=torch.float32)
                pose6d_validator.add_result({
                    "gt_class_ids": t_label.numpy(), # shape: (n_positive,)
                    "gt_scales": gt_size, # shape: list of lists, each list is [w, h, d] * n_positive
                    "gt_RTs": gt_pose, # shape: list of 4x4 matrices, each is a pose 
                    "pred_class_ids": labels.numpy(), # shape: (topk,)
                    "pred_scales": pred_size, # list of lists, each list is [w, h, d] * topk
                    "pred_RTs": pred_pose, # list of 4x4 matrices, each is a pose * topk
                    "pred_scores": scores.numpy() # shape: (topk,)
                })
                
                draw_meta_info['cls_type'].append(target_label_id)
                draw_meta_info['pred_pose'].append(pred_pose)
                draw_meta_info['pred_size'].append(pred_size)
                draw_meta_info['gt_pose'].append(gt_pose)

            if args.store_image:
                # load image
                img_fname = os.path.join(args.root_dir, seq_id, frame_id + '.jpg')
                img = cv2.imread(img_fname)
                if img is None:
                    print('Image not found: {}'.format(img_fname))
                    continue
                PL = pred_dict[cls_type][seq_id][frame_id]["P_left"]
                PR = pred_dict[cls_type][seq_id][frame_id]["P_right"]
                PL = torch.tensor(PL, dtype=torch.float32)
                PR = torch.tensor(PR, dtype=torch.float32)
                
                h, w, _ = img.shape
                img_l = img[:, :w//2, :].astype(np.uint8).copy()
                img_r = img[:, w//2:, :].astype(np.uint8).copy()
                
                # Draw all predictions and ground truth from draw_meta_info
                for cls_idx, cls_name in enumerate(draw_meta_info['cls_type']):
                    pred_poses = draw_meta_info['pred_pose'][cls_idx]
                    gt_poses = draw_meta_info['gt_pose'][cls_idx]
                    label_id = draw_meta_info['cls_type'][cls_idx]
                    
                    # Draw predictions
                    color = colors_rgb[label_id % len(colors_rgb)]
                    for pred_idx, pred_pose_matrix in enumerate(pred_poses):
                        pred_size = draw_meta_info['pred_size'][cls_idx]
                        canonical_box = get_canonical_box(pred_size[0])
                        R, t = pred_pose_matrix[:3, :3], pred_pose_matrix[:3, 3]
                        R, t = torch.tensor(R, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)
                        box = (R @ canonical_box.T).T + t
                        box_l = project_3d_to_2d_batch(torch.tensor(box), PL).numpy()
                        box_r = project_3d_to_2d_batch(torch.tensor(box), PR).numpy()
                        
                        draw_box(box_l, box_r, img_l, img_r, color=color)

                img_l = cv2.resize(img_l, (640, 640))
                img_r = cv2.resize(img_r, (640, 640))
                combined_img = np.concatenate([img_l, img_r], axis=1)
                
                save_dir = os.path.join(args.out_dir, seq_id)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{seq_id}_{frame_id}.jpg')
                cv2.imwrite(save_path, combined_img)

    pose6d_validator.compute_metrics()
    mean_res, class_res = pose6d_validator.get_result()
    # Store results in JSON format
    result_output = {
        "mean": mean_res,
    }
    for cls, res in class_res.items():
        result_output[cls] = res

    output_json_path = os.path.join(args.out_dir, 'pose6d_evaluation_results.json')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(result_output, f, indent=4)

    print(f"Results saved to: {output_json_path}")
