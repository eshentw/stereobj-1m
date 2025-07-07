'''
    Single-GPU inference on val or test set.
'''
import argparse
import cv2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import torch
import importlib
import os
import sys
import h5py
import tqdm
from omegaconf import OmegaConf

from pose6d_validator import Pose6DValidator
from datasets.utils import kpts_to_boxes, box_cxcywh_to_xyxy, project_3d_to_2d_batch, \
    kpts_pose_disp_to_left_right, gen_reproj_matrix_batch, reproject_2d_to_3d_batch
from datasets.box import Box, procrustes_alignment, get_canonical_box
from datasets import build_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, '..', 'data_loader'))

from dict_restore import DictRestore
from saver_restore import SaverRestore
# import triangulation_object



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_res34_backbone', help='Model name [default: model_res34_backbone]')
parser.add_argument('--model_path', default=None, help='Model checkpint path [default: ]')
parser.add_argument('--split', default='val', help='Dataset split [default: test]')
parser.add_argument('--dataset', default='stereobj1m_dataset', help='Dataset name [default: stereobj1m_dataset]')
parser.add_argument('--num_kp', type=int, default=64, help='Number of Keypoints [default: 1024]')
parser.add_argument('--num_workers', type=int, default=4, help='Number of multiprocessing workers [default: 1024]')
parser.add_argument('--image_width', type=int, default=768, help='Image width [default: 768]')
parser.add_argument('--image_height', type=int, default=768, help='Image height [default: 768]')
parser.add_argument('--data', default='', help='Data path [default: ]')
parser.add_argument('--cls_type', default='', help='Object class of interest [default: ]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--command_file', default=None, help='Command file name [default: None]')
parser.add_argument('--debug', type=int, default=0, help='Debug mode [default: 0]')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
dataset = importlib.import_module(args.dataset)

BATCH_SIZE = args.batch_size
GPU_INDEX = args.gpu

MODEL = importlib.import_module(args.model) # import network module
MODEL_FILE = os.path.join(args.model+'.py')

cfg = dict(
        dataset='stereobj',
        root='data/stereobj_1m/images_annotations',
        names={
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
        },
        kpt_type=8,
        n_kpts=args.num_kp,
        random_size_crop=[0.8, 1.0],
        resize=[args.image_height, args.image_width],
        scale_jitter=[0.95, 1.05],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_obj=args.cls_type,
    )
cfg = OmegaConf.create(cfg)
# dataset
test_data_loader = build_dataset(cfg, args.split)

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
target_label_id = class_to_label.get(args.cls_type, None)

if args.cls_type.startswith('sterile_tip_rack'):
    target_label_id = class_to_label.get('sterile_tip_rack', None)
elif args.cls_type.startswith('tube_rack'):
    target_label_id = class_to_label.get('tube_rack', None)
elif args.cls_type.startswith('pipette'):
    target_label_id = class_to_label.get('pipette', None)
assert target_label_id is not None, f"Class {args.cls_type} not found in dataset classes."
classes = [args.cls_type]

pose6d_validator = Pose6DValidator(classes, use_matches_for_pose=True)

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            image_pl, labels_pl = MODEL.placeholder_inputs( \
                    args.batch_size, args.image_height, args.image_width,
                    args.num_kp, debug=args.debug)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)

            # Get model and loss
            end_points = MODEL.get_model(image_pl, args.num_kp, \
                    is_training=is_training_pl, debug=args.debug)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        if args.model_path is not None:
            if 'npz' in args.model_path:
                dict_file = np.load(args.model_path)
                dict_for_restore = {}
                dict_file_keys = dict_file.keys()
                for k in dict_file_keys:
                    dict_for_restore[k] = dict_file[k]
                dict_for_restore = MODEL.name_mapping(dict_for_restore, debug=args.debug)
                dr = DictRestore(dict_for_restore, print)
                dr.run_init(sess)
                print("npz file restored.")
            elif '.h5' in args.model_path:
                f = h5py.File(args.model_path, 'r')
                dict_for_restore = {}
                for k in f.keys():
                    for group in f[k].items():
                        for g in group[1]:
                            dict_for_restore[os.path.join(k, g)] = group[1][g][:]
                            value = group[1][g][:]
                dict_for_restore = MODEL.name_mapping(dict_for_restore, debug=args.debug)
                dr = DictRestore(dict_for_restore, print)
                dr.run_init(sess)
                print("h5 file restored.")
            else:
                sr = SaverRestore(args.model_path, print) #, ignore=['batch:0'])
                sr.run_init(sess)
                print("Model restored.")

        ops = {'image_pl': image_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'end_points': end_points,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    save_dir = os.path.join('log_lr_{}_preds'.format(args.split), args.cls_type)
    if not os.path.exists(save_dir):
        os.system('mkdir -p {}'.format(save_dir))

    for idx, batch in enumerate(tqdm.tqdm(test_data_loader)):
        sample, targets = batch
        image_l = sample.image_l.numpy().transpose(0, 2, 3, 1)
        image_r = sample.image_r.numpy().transpose(0, 2, 3, 1)

        feed_dict = {ops['image_pl']: image_l,
                     ops['is_training_pl']: is_training,}

        pred_kp_uv_val_l = sess.run(ops['end_points']['pred_kp_uv'], feed_dict=feed_dict)

        feed_dict = {ops['image_pl']: image_r,
                     ops['is_training_pl']: is_training,}

        pred_kp_uv_val_r = sess.run(ops['end_points']['pred_kp_uv'], feed_dict=feed_dict)

        pred_kp_uv_val_l = pred_kp_uv_val_l[:, :16]
        pred_kp_uv_val_r = pred_kp_uv_val_r[:, :16]
        
        targs_l, targs_r, tart_kpts_l, tart_kpts_r = [], [], [], []
        targs_nb_kpts_2d_l,  targs_nb_kpts_2d_r = [], []
        for i in range(len(pred_kp_uv_val_l)):
            t_label = targets["labels"][i]
            if target_label_id not in t_label:
                continue
            label_index = (t_label == target_label_id).nonzero(as_tuple=True)[0][0].item()
            t_label = torch.tensor([target_label_id])

            t_box_l = targets["boxes_l"][i][label_index].unsqueeze(0)
            t_box_r = targets["boxes_r"][i][label_index].unsqueeze(0)
            t_kpts_l, t_kpts_r = kpts_pose_disp_to_left_right(
                targets["kpts_pose"][i][label_index], targets["kpts_disp"][i][label_index])

            t_kpts_vis = targets["kpts_pose"][i][label_index][..., 2:]
            t_kpts_l = torch.cat([t_kpts_l, t_kpts_vis], dim=-1).unsqueeze(0)
            t_kpts_r = torch.cat([t_kpts_r, t_kpts_vis], dim=-1).unsqueeze(0)

            t_box_l, t_box_r = t_box_l.cpu(), t_box_r.cpu()
            t_kpts_l, t_kpts_r = t_kpts_l.cpu(), t_kpts_r.cpu()
            t_label = t_label.cpu()

            # convert to [x0, y0, x1, y1] format and to absolute coordinates
            # so that it matches the format of predicted boxes
            t_box_l = box_cxcywh_to_xyxy(t_box_l)
            t_box_r = box_cxcywh_to_xyxy(t_box_r)

            img_h, img_w = targets["ori_shape"][i]
            scale_fct = torch.as_tensor([img_w, img_h, img_w, img_h])
            t_box_l = t_box_l * scale_fct[None, :]
            t_box_r = t_box_r * scale_fct[None, :]

            scale_fct = torch.as_tensor([img_w, img_h, 1])
            t_kpts_l = t_kpts_l * scale_fct[None, None, :]
            t_kpts_r = t_kpts_r * scale_fct[None, None, :]

            targs_l.append(torch.cat(
                [t_box_l, t_label[:, None].float()], dim=-1))
            targs_r.append(torch.cat(
                [t_box_r, t_label[:, None].float()], dim=-1))

            tart_kpts_l.append(t_kpts_l)
            tart_kpts_r.append(t_kpts_r)
            
            gt_nb_kpts_3d = targets["nb_kpts_3d"][i][label_index].to(torch.float32).cpu()
            target_kpts_l = targets["nb_kpts_2d_l"][i][label_index].to(torch.float32).cpu()
            target_kpts_r = targets["nb_kpts_2d_r"][i][label_index].to(torch.float32).cpu()

            targs_nb_kpts_2d_l.append(target_kpts_l)
            targs_nb_kpts_2d_r.append(target_kpts_r)
            # keep only the top 50 to save time for validation
            # all_scores = preds_l[i][:, -2].cpu()
            # k = min(50, all_scores.size(0))
            # scores, topk_idx = torch.topk(all_scores, k)
            # labels = preds_l[i][topk_idx, -1].long().cpu()
            labels = t_label.long().cpu()

            # 2D predicted keypoints
            k2d_l = torch.tensor(pred_kp_uv_val_l[i]).unsqueeze(0)
            k2d_r = torch.tensor(pred_kp_uv_val_r[i]).unsqueeze(0)

            # 3D reproject
            PL = targets['proj_matrix_l'][i].to(torch.float32).cpu()
            PR = targets['proj_matrix_r'][i].to(torch.float32).cpu()
            baseline = targets['baseline'][i].to(torch.float32).cpu()
            gt_k3d = targets['kpts_3d'][i][label_index].cpu().unsqueeze(0)
            Q = gen_reproj_matrix_batch(PL, PR, baseline)
            k3d = reproject_2d_to_3d_batch(k2d_l, k2d_r, Q)
            pred_size, gt_size = [], []
            pred_pose, gt_pose = [], []

            img_h, img_w = targets["ori_shape"][i]
            for pd, gt in zip(k3d, gt_k3d):
                # gt
                box = Box.from_keypoints(gt)
                R_gt, t_gt = box.get_pose()
                gt_pose.append(torch.cat((R_gt, t_gt.reshape(3, 1)), dim=1))
                gt_size.append(box.get_size().tolist())
                # prediction
                R, t = procrustes_alignment(gt_nb_kpts_3d, pd)
                pred_pose.append(torch.cat((R, t.reshape(3, 1)), dim=1))
                pred_size.append(box.get_size().tolist())
            # Convert pred_pose from 3x4 to 4x4
            pred_pose = [
                torch.cat((pose, torch.tensor([[0, 0, 0, 1]])), dim=0).numpy()
                for pose in pred_pose]
            gt_pose = [
                torch.cat((pose, torch.tensor([[0, 0, 0, 1]])), dim=0).numpy()
                for pose in gt_pose]

            # pose6d_validator.add_result({
            #     "gt_class_ids": t_label.numpy(), # shape: (n_positive,)
            #     "gt_scales": gt_size, # shape: list of lists, each list is [w, h, d] * n_positive
            #     "gt_RTs": gt_pose, # shape: list of 4x4 matrices, each is a pose 
            #     "pred_class_ids": labels.numpy(),
            #     "pred_scales": pred_size,
            #     "pred_RTs": pred_pose,
            #     "pred_scores": scores.numpy()
            # })

        view = True
        if view:
            pred_kp_uv_show_l = pred_kp_uv_val_l[0]
            pred_kp_uv_show_r = pred_kp_uv_val_r[0]
            PL = targets['proj_matrix_l'][0].to(torch.float32).cpu()
            PR = targets['proj_matrix_r'][0].to(torch.float32).cpu()
            R, t = torch.tensor(pred_pose[0][:3, :3]).to(torch.float32), \
                torch.tensor(pred_pose[0][:3, 3]).to(torch.float32)
            pred_size = pred_size[0]
            canonical_box = get_canonical_box(pred_size)
            box = (R @ canonical_box.T).T + t
            box_l = project_3d_to_2d_batch(box, PL).numpy()
            box_r = project_3d_to_2d_batch(box, PR).numpy()

            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_show_l = image_l[0]
            image_show_r = image_r[0]
            image_show_l = image_show_l * std + mean
            image_show_r = image_show_r * std + mean
            image_show_l_cv2 = (image_show_l * 255).astype(np.uint8)
            image_show_r_cv2 = (image_show_r * 255).astype(np.uint8)
            img_l_vis = image_show_l_cv2.copy()
            img_r_vis = image_show_r_cv2.copy()
            
            # Draw predicted keypoints in red
            for i in range(len(pred_kp_uv_show_l)):
                pt_l = (int(pred_kp_uv_show_l[i, 0]), int(pred_kp_uv_show_l[i, 1]))
                cv2.circle(img_l_vis, pt_l, 2, (0, 0, 255), -1)
                cv2.putText(img_l_vis, f'p{i}', (pt_l[0]+5, pt_l[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            for i in range(len(pred_kp_uv_show_r)):
                pt_r = (int(pred_kp_uv_show_r[i, 0]), int(pred_kp_uv_show_r[i, 1]))
                cv2.circle(img_r_vis, pt_r, 2, (0, 0, 255), -1)
                cv2.putText(img_r_vis, f'p{i}', (pt_r[0]+5, pt_r[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw target nb keypoints in blue
            if len(targs_nb_kpts_2d_l) > 0:
                target_kpts_l = targs_nb_kpts_2d_l[0]
                for i in range(len(target_kpts_l)):
                    pt_l = (int(target_kpts_l[i, 0]), int(target_kpts_l[i, 1]))
                    cv2.circle(img_l_vis, pt_l, 2, (255, 0, 0), -1)
                    cv2.putText(img_l_vis, f't{i}', (pt_l[0]+5, pt_l[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            if len(targs_nb_kpts_2d_r) > 0:
                target_kpts_r = targs_nb_kpts_2d_r[0]
                for i in range(len(target_kpts_r)):
                    pt_r = (int(target_kpts_r[i, 0]), int(target_kpts_r[i, 1]))
                    cv2.circle(img_r_vis, pt_r, 2, (255, 0, 0), -1)
                    cv2.putText(img_r_vis, f't{i}', (pt_r[0]+5, pt_r[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Draw 3D box projections in green with lines connecting the box edges
            if len(box_l) >= 8:
                box_edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                    (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
                ]
                
                # Draw box points and edges for left image
                for i in range(len(box_l)):
                    pt_l = (int(box_l[i, 0]), int(box_l[i, 1]))
                    cv2.circle(img_l_vis, pt_l, 4, (0, 255, 0), -1)
                    cv2.putText(img_l_vis, f'b{i}', (pt_l[0]+5, pt_l[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Draw edges for left image
                for edge in box_edges:
                    pt1_l = (int(box_l[edge[0], 0]), int(box_l[edge[0], 1]))
                    pt2_l = (int(box_l[edge[1], 0]), int(box_l[edge[1], 1]))
                    cv2.line(img_l_vis, pt1_l, pt2_l, (0, 255, 0), 2)
                
                # Draw box points and edges for right image
                for i in range(len(box_r)):
                    pt_r = (int(box_r[i, 0]), int(box_r[i, 1]))
                    cv2.circle(img_r_vis, pt_r, 4, (0, 255, 0), -1)
                    cv2.putText(img_r_vis, f'b{i}', (pt_r[0]+5, pt_r[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Draw edges for right image
                for edge in box_edges:
                    pt1_r = (int(box_r[edge[0], 0]), int(box_r[edge[0], 1]))
                    pt2_r = (int(box_r[edge[1], 0]), int(box_r[edge[1], 1]))
                    cv2.line(img_r_vis, pt1_r, pt2_r, (0, 255, 0), 2)
            else:
                # Fallback: just draw points if we don't have enough for a full box
                for i in range(len(box_l)):
                    pt_l = (int(box_l[i, 0]), int(box_l[i, 1]))
                    cv2.circle(img_l_vis, pt_l, 4, (0, 255, 0), -1)
                    cv2.putText(img_l_vis, f'b{i}', (pt_l[0]+5, pt_l[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                for i in range(len(box_r)):
                    pt_r = (int(box_r[i, 0]), int(box_r[i, 1]))
                    cv2.circle(img_r_vis, pt_r, 4, (0, 255, 0), -1)
                    cv2.putText(img_r_vis, f'b{i}', (pt_r[0]+5, pt_r[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Concatenate images horizontally
            combined_img = np.concatenate([img_l_vis, img_r_vis], axis=1)
            
            # Display using cv2
            cv2.imshow('Left and Right Images with Keypoints', combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    train()
