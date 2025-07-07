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
parser.add_argument('--object_data', default='', help='Object data directory [default: ]')
parser.add_argument('--input_json', default='', help='Input json filename [default: ]')
parser.add_argument('--out_dir', default='', help='Output directory [default: ]')
parser.add_argument('--gt_json', default='', help='GT json filename [default: ]')
parser.add_argument('--root_dir', default='', help='Root directory for data [default: ]')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


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

    class_to_label = {cls: idx for idx, cls in enumerate(classes)}
    label_to_class = {idx: cls for idx, cls in enumerate(classes)}

    pose6d_validator = Pose6DValidator(classes, use_matches_for_pose=True)
    with open(args.input_json, 'r') as f:
        input_dict = json.load(f)
    pred_dict = input_dict['pred']

    with open(args.gt_json, 'r') as f:
        gt_dict = json.load(f)

    assert input_dict['split'] == gt_dict['split']
    gt_dict = gt_dict['pred']

    result_dict = {}
    cls_count = Counter()

    for cls_type in gt_dict:
        obj_points_fname = os.path.join(args.object_data, cls_type + '.xyz')
        with open(obj_points_fname, 'r') as f:
            data = f.read().rstrip().split()
            data = [float(d) for d in data]
        target_label_id = class_to_label.get(cls_type, None)

        if cls_type.startswith('sterile_tip_rack'):
            target_label_id = class_to_label.get('sterile_tip_rack', None)
        elif cls_type.startswith('tube_rack'):
            target_label_id = class_to_label.get('tube_rack', None)
        elif cls_type.startswith('pipette'):
            target_label_id = class_to_label.get('pipette', None)
        assert target_label_id is not None, f"Class {cls_type} not found in dataset classes."
        print('Evaluating class: {}'.format(cls_type))

        for seq_id in gt_dict[cls_type]:
            for frame_id in tqdm(gt_dict[cls_type][seq_id], desc=f"Processing {seq_id}"):
                try:
                    assert seq_id in pred_dict[cls_type]
                    assert frame_id in pred_dict[cls_type][seq_id]
                except:
                    # print('ERROR in {}'.format(seq_id + ',' + frame_id))
                    continue

                pose_pred = pred_dict[cls_type][seq_id][frame_id]["pose_pred"]
                pose_pred = torch.tensor(pose_pred, dtype=torch.float32)
                if pose_pred.ndim == 2:
                    pose_pred = pose_pred.unsqueeze(0)

                pose_gt = gt_dict[cls_type][seq_id][frame_id]
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
                
                view = False
                if view:
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

                    canonical_box = get_canonical_box(pred_size)
                    R, t = torch.tensor(pred_pose[0][:3, :3]).to(torch.float32), \
                        torch.tensor(pred_pose[0][:3, 3]).to(torch.float32)
                    box = (R @ canonical_box.T).T + t
                    box_l = project_3d_to_2d_batch(box, PL).numpy()
                    box_r = project_3d_to_2d_batch(box, PR).numpy()

                    # mean = np.array([0.485, 0.456, 0.406])
                    # std = np.array([0.229, 0.224, 0.225])
                    # img_l = img[:, :w//2, :] * std + mean
                    # img_r = img[:, w//2:, :] * std + mean
                    h, w, _ = img.shape
                    img_l = img[:, :w//2, :]
                    img_r = img[:, w//2:, :]
                    img_l = img_l.astype(np.uint8).copy()
                    img_r = img_r.astype(np.uint8).copy()
                    
                    if len(box_l) >= 8:
                        box_edges = [
                            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
                        ]
                        
                        # Draw box points and edges for left image
                        for i in range(len(box_l)):
                            pt_l = (int(box_l[i, 0]), int(box_l[i, 1]))
                            cv2.circle(img_l, pt_l, 4, (0, 255, 0), -1)
                            cv2.putText(img_l, f'b{i}', (pt_l[0]+5, pt_l[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
                        # Draw edges for left image
                        for edge in box_edges:
                            pt1_l = (int(box_l[edge[0], 0]), int(box_l[edge[0], 1]))
                            pt2_l = (int(box_l[edge[1], 0]), int(box_l[edge[1], 1]))
                            cv2.line(img_l, pt1_l, pt2_l, (0, 255, 0), 2)
                        
                        # Draw box points and edges for right image
                        for i in range(len(box_r)):
                            pt_r = (int(box_r[i, 0]), int(box_r[i, 1]))
                            cv2.circle(img_r, pt_r, 4, (0, 255, 0), -1)
                            cv2.putText(img_r, f'b{i}', (pt_r[0]+5, pt_r[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
                        # Draw edges for right image
                        for edge in box_edges:
                            pt1_r = (int(box_r[edge[0], 0]), int(box_r[edge[0], 1]))
                            pt2_r = (int(box_r[edge[1], 0]), int(box_r[edge[1], 1]))
                            cv2.line(img_r, pt1_r, pt2_r, (0, 255, 0), 2)
                    else:
                        # Fallback: just draw points if we don't have enough for a full box
                        for i in range(len(box_l)):
                            pt_l = (int(box_l[i, 0]), int(box_l[i, 1]))
                            cv2.circle(img_l, pt_l, 4, (0, 255, 0), -1)
                            cv2.putText(img_l, f'b{i}', (pt_l[0]+5, pt_l[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
                        for i in range(len(box_r)):
                            pt_r = (int(box_r[i, 0]), int(box_r[i, 1]))
                            cv2.circle(img_r, pt_r, 4, (0, 255, 0), -1)
                            cv2.putText(img_r, f'b{i}', (pt_r[0]+5, pt_r[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    img_l = cv2.resize(img_l, (640, 640))
                    img_r = cv2.resize(img_r, (640, 640))
                    combined_img = np.concatenate([img_l, img_r], axis=1)
                    cv2.imshow('Left and Right Images with Keypoints', combined_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            pose6d_validator.compute_metrics()
            res = pose6d_validator.get_result()
            
            class_name = label_to_class[target_label_id]
            if class_name not in result_dict:
                result_dict[class_name] = res
            else:
                # accum the cls counter
                cls_count[class_name] += 1
                
                for key in res:
                    if key in result_dict[class_name]:
                        result_dict[class_name][key] += res[key]
                    else:
                        result_dict[class_name][key] = res[key]

            print(f'Results for {cls_type}: {res}')
            pose6d_validator.init_metrics()

    # average the results
    for class_name in result_dict:
        for key in result_dict[class_name]:
            if class_name in cls_count and cls_count[class_name] > 0:
                result_dict[class_name][key] /= cls_count[class_name]
            else:
                result_dict[class_name][key] = 0.0

    output_json_path = os.path.join(args.out_dir, 'evaluation_results.json')
    with open(output_json_path, 'w') as f:
        json.dump(result_dict, f, indent=4)

                    





