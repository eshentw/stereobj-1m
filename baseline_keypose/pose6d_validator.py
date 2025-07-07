# Modified from https://github.com/mentian/object-deformnet

import math
import numpy as np

_SYMMETRIC_OBJECTS = [
    'bottle', 'cup', 'ball', 'screwdriver', 'centrifuge_tube'
]


def get_3d_bbox_batch(sizes, shifts):
    """
    Args:
        sizes: (M, 3) array of box dimensions [dx, dy, dz]
        shifts: (M, 3) array of box centers

    Returns:
        corners: (M, 8, 3) array of each box's 8 corner coordinates
    """
    sizes = np.asarray(sizes, dtype=float).reshape(-1, 3)
    shifts = np.asarray(shifts, dtype=float).reshape(-1, 3)

    # Precompute the 8 "corner signs" once.
    signs = np.array(
        [[sx, sy, sz]
         for sx in (1, -1)
         for sy in (1, -1)
         for sz in (1, -1)]
    ) * 0.5  # shape (8, 3)

    # Broadcast signs (1, 8, 3) against sizes (M, 1, 3) and shifts (M, 1, 3).
    return signs[None, :, :] * sizes[:, None, :] + shifts[:, None, :]


def transform_batch(points, sRT):
    """
    Args:
        points: (M, 3, N) array of N points per batch
        sRT:    (M, 4, 4) array of transforms

    Returns:
        transformed: (M, 3, N) array of transformed points
    """
    points = np.asarray(points, dtype=float)
    sRT = np.asarray(sRT, dtype=float)
    M, _, N = points.shape

    # Make homogeneous coords: (M, 4, N).
    hom = np.concatenate(
        [points, np.ones((M, 1, N), dtype=points.dtype)],
        axis=1,
    )
    # Batch-matmul: (M, 4, N) = (M, 4, 4) @ (M, 4, N).
    out = np.matmul(sRT, hom)
    # Dehomogenize back to (M, 3, N).
    return out[:, :3, :] / out[:, 3:4, :]


def compute_3d_iou_batch(sRT1, sRT2, size1, size2, class_id, synset_names):
    """
    Args:
        sRT1:  (M, 4, 4) array of box poses
        sRT2:  (N, 4, 4) array of box poses
        size1: (M, 3) array of [dx, dy, dz]
        size2: (N, 3) array of [dx, dy, dz]

    Returns:
        ious:  (M, N) pairwise IoU matrix
    """
    sRT1 = np.asarray(sRT1, dtype=float)
    sRT2 = np.asarray(sRT2, dtype=float)
    size1 = np.asarray(size1, dtype=float).reshape(-1, 3)
    size2 = np.asarray(size2, dtype=float).reshape(-1, 3)

    def asymmetric_3d_iou(sRT1_i, sRT2_i, size1_i, size2_i):
        """
        Compute IoU by treating boxes as axis-aligned bounding volumes.
        """
        shift1 = np.zeros_like(size1_i)
        shift2 = np.zeros_like(size2_i)
        corners1 = get_3d_bbox_batch(size1_i, shift1)
        corners2 = get_3d_bbox_batch(size2_i, shift2)

        # Transform to world coords: (M, 3, 8) & (N, 3, 8).
        pts1 = transform_batch(corners1.transpose(0, 2, 1), sRT1_i)
        pts2 = transform_batch(corners2.transpose(0, 2, 1), sRT2_i)

        mn1 = pts1.min(axis=1)
        mx1 = pts1.max(axis=1)
        mn2 = pts2.min(axis=1)
        mx2 = pts2.max(axis=1)

        overlap_min = np.maximum(mn1[:, None, :], mn2[None, :, :])
        overlap_max = np.minimum(mx1[:, None, :], mx2[None, :, :])
        delta = overlap_max - overlap_min

        inter = np.prod(np.clip(delta, a_min=0.0, a_max=None), axis=2)
        vol1 = np.prod(mx1 - mn1, axis=1)
        vol2 = np.prod(mx2 - mn2, axis=1)

        union = vol1[:, None] + vol2[None, :] - inter
        return inter / union

    def x_rotation_matrix(theta):
        """Return 4x4 rotation matrix around the X axis."""
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array(
            [
                [1.0,   0.0,    0.0,   0.0],
                [0.0, cos_t, -sin_t,   0.0],
                [0.0, sin_t,  cos_t,   0.0],
                [0.0,   0.0,    0.0,   1.0],
            ],
            dtype=float,
        )

    # sample n rotations around X and compute all IoUs
    n = 20
    thetas = np.linspace(0, 2 * math.pi, n, endpoint=False)
    all_ious = []
    for t in thetas:
        R = x_rotation_matrix(t)
        # rotate every M pose by R (broadcast matmul)
        rotated_sRT1 = sRT1 @ R           # still shape (M,4,4)
        iou_mat = asymmetric_3d_iou(rotated_sRT1, sRT2, size1, size2)
        all_ious.append(iou_mat)

    # stack to (n, M, N) and take elementwise max → (M,N)
    all_ious = np.stack(all_ious, axis=0)
    symm_ious = all_ious.max(axis=0)

    # asymmetric IoU
    asym_ious = asymmetric_3d_iou(sRT1, sRT2, size1, size2)

    num_p, num_g = sRT1.shape[0], sRT2.shape[0]
    ious = np.zeros((num_p, num_g), dtype=np.float32)
    for i in range(num_p):
        for j in range(num_g):
            cls_name = synset_names[class_id[i]]
            if cls_name in _SYMMETRIC_OBJECTS:
                ious[i, j] = symm_ious[i, j]
            else:
                ious[i, j] = asym_ious[i, j]

    return ious


def compute_rt_error_batch(sRT1, sRT2, class_id, synset_names):
    """
    Args:
      sRT1:        (M,4,4) array of source poses
      sRT2:        (N,4,4) array of target poses
      symmetric_y: if True, measure rotation only around the world-Y axis

    Returns:
      errors: (M,N,2) array where
        errors[i,j,0] = rotation error (degrees)
        errors[i,j,1] = translation error (cm)
    """
    sRT1 = np.asarray(sRT1, float)
    sRT2 = np.asarray(sRT2, float)

    # Extract R & T
    R1 = sRT1[:, :3, :3]  # (M,3,3)
    T1 = sRT1[:, :3,  3]  # (M,3)
    R2 = sRT2[:, :3, :3]  # (N,3,3)
    T2 = sRT2[:, :3,  3]  # (N,3)

    # Remove scale
    det1 = np.linalg.det(R1)  # (M,)
    det2 = np.linalg.det(R2)  # (N,)
    R1 = R1 / np.cbrt(det1)[:, None, None]
    R2 = R2 / np.cbrt(det2)[:, None, None]

    # Translation errors (cm)
    diff = T1[:, None, :] - T2[None, :, :]       # (M,N,3)
    shifts = np.linalg.norm(diff, axis=2) * 100   # (M,N)

    # --- angle around X-axis only ---
    y = np.array([1.0, 0.0, 0.0])
    # Project each R onto y
    y1 = R1 @ y  # (M,3)
    y2 = R2 @ y  # (N,3)
    # Norms
    n1 = np.linalg.norm(y1, axis=1)  # (M,)
    n2 = np.linalg.norm(y2, axis=1)  # (N,)
    # Dot products & broadcast to (M,N)
    dots = np.einsum('md,nd->mn', y1, y2)
    cos_theta = dots / (n1[:, None] * n2[None, :])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    symm_angles = np.degrees(np.arccos(cos_theta))  # (M,N)

    # --- Full rotation difference via trace formula ---
    R2_T = R2.transpose(0, 2, 1)         # (N,3,3)
    R_diff = R1[:, None] @ R2_T[None]    # (M,N,3,3)
    traces = np.trace(R_diff, axis1=2, axis2=3)  # (M,N)
    cos_theta = (traces - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    asym_angles = np.degrees(np.arccos(cos_theta))  # (M,N)

    num_p, num_g = sRT1.shape[0], sRT2.shape[0]
    angles = np.zeros((num_p, num_g), dtype=np.float32)
    for i in range(num_p):
        for j in range(num_g):
            cls_name = synset_names[class_id[j]]

            if cls_name in _SYMMETRIC_OBJECTS:
                angles[i, j] = symm_angles[i, j]
            else:
                angles[i, j] = asym_angles[i, j]

    # stack results
    errors = np.stack([angles, shifts], axis=2)  # (M,N,2)
    return errors


def match_iou(gt_ids, gt_RT, gt_sz, pr_ids, pr_RT, pr_sz, pr_scores,
              iou_ths, score_th, synset_names):
    """Match predictions to GT at multiple IoU thresholds."""
    num_p, num_g = len(pr_ids), len(gt_ids)
    indices = np.argsort(pr_scores)[::-1]
    pr_ids, pr_RT, pr_sz, pr_scores = \
        pr_ids[indices], pr_RT[indices], pr_sz[indices], pr_scores[indices]

    num_iou_3d_thres = len(iou_ths)
    pm = -1 * np.ones([num_iou_3d_thres, num_p])
    gm = -1 * np.ones([num_iou_3d_thres, num_g])

    # nothing to match
    if num_p == 0 or num_g == 0:
        return gm, pm, indices

    # (num_p, num_g)
    overlaps = compute_3d_iou_batch(
        pr_RT, gt_RT, pr_sz, gt_sz, pr_ids, synset_names)

    # loop through predictions and find matching ground truth boxes
    for s, iou_thres in enumerate(iou_ths):
        for i in range(indices.shape[0]):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(overlaps[i, sorted_ixs] < score_th)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                if gm[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                if iou < iou_thres:
                    break
                if iou > iou_thres:
                    gm[s, j] = i
                    pm[s, i] = j
                    break

    return gm, pm, indices


def match_rt(pr_ids, pr_RT, gt_ids, gt_RT, deg_ths, shift_ths, synset_names):
    """Match predictions to GT based on pose error thresholds."""
    num_p, num_g = len(pr_ids), len(gt_ids)
    pm = -np.ones((len(deg_ths) + 1, len(shift_ths) + 1, num_p), int)
    gm = -np.ones((len(deg_ths) + 1, len(shift_ths) + 1, num_g), int)
    # nothing to match
    if num_p == 0 or num_g == 0:
        return gm, pm

    # compute pose errors: shape (num_p, num_g, 2)
    errors = compute_rt_error_batch(pr_RT, gt_RT, pr_ids, synset_names)

    for di, d in enumerate(deg_ths + [360]):
        for si, s in enumerate(shift_ths + [100]):
            for i in range(num_p):
                order = np.argsort(errors[i, :, 0] + errors[i, :, 1])
                for j in order:
                    # If ground truth box is already matched, go to next one
                    if gm[di, si, j] >= 0:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if errors[i, j, 0] > d or errors[i, j, 1] > s:
                        continue
                    pm[di, si, i] = j
                    gm[di, si, j] = i
                    break
    return gm, pm


def compute_ap_acc(pred_matches, pred_scores, gt_matches):
    """Compute AP and accuracy given matches (>=0) and scores."""
    # sort the scores from high to low
    assert pred_matches.shape[0] == pred_scores.shape[0]
    order = np.argsort(pred_scores)[::-1]
    m = pred_matches[order]
    tp = np.cumsum(m >= 0)
    # Pad with start and end values to simplify the math
    prec = np.concatenate([[0], tp / (np.arange(len(m)) + 1), [0]])
    rec = np.concatenate([[0], tp / len(gt_matches), [1]])
    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    # compute mean AP over recall range
    idx = np.where(rec[:-1] != rec[1:])[0] + 1
    ap = np.sum((rec[idx] - rec[idx - 1]) * prec[idx])
    acc = tp[-1] / len(m) if len(m) > 0 else 0
    return ap, acc


class Pose6DValidator:
    """Class to validate 6D pose estimation results."""

    def __init__(self, synset_names, use_matches_for_pose=False):
        self.synset_names = synset_names
        self.use_matches_for_pose = use_matches_for_pose

        self.degree_thres = list(range(0, 61, 1))
        self.shift_thres = [i / 2 for i in range(21)]
        self.iou_thres = [i / 100 for i in range(101)]
        self.iou_pose_thres = 0.1

        if use_matches_for_pose:
            assert self.iou_pose_thres in self.iou_thres

        self.classes = ['mean'] + synset_names
        self.init_metrics()

    def init_metrics(self):
        C = len(self.classes)

        # containers
        self.iou_ap = np.zeros((C, len(self.iou_thres)))
        self.iou_acc = np.zeros_like(self.iou_ap)
        self.pose_ap = np.zeros(
            (C, len(self.degree_thres) + 1, len(self.shift_thres) + 1))
        self.pose_acc = np.zeros_like(self.pose_ap)

        self.data = {c: {
            'iou': {'pm': [], 'scores': [], 'gm': []},
            'pose': {'pm': [], 'scores': [], 'gm': []}
        } for c in range(C)}

    def add_result(self, result):
        """Add a single prediction result."""
        assert isinstance(result, dict), "Result must be a dictionary."
        required_keys = ['gt_class_ids', 'gt_RTs', 'gt_scales',
                         'pred_class_ids', 'pred_RTs', 'pred_scales',
                         'pred_scores']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        gt_ids = result['gt_class_ids'].astype(int)
        gt_RT = np.array(result['gt_RTs'])
        gt_sz = np.array(result['gt_scales'])
        pr_ids = result['pred_class_ids'].astype(int)
        pr_RT = np.array(result['pred_RTs'])
        pr_sz = np.array(result['pred_scales'])
        pr_sc = result['pred_scores']

        if not (len(gt_ids) or len(pr_ids)):
            return

        for cid in range(1, len(self.classes)):
            m_gt = gt_ids == (cid - 1)
            m_pr = pr_ids == (cid - 1)
            g_ids = gt_ids[m_gt]
            g_RT = gt_RT[m_gt]
            g_sz = gt_sz[m_gt]
            p_ids = pr_ids[m_pr]
            p_RT = pr_RT[m_pr]
            p_sz = pr_sz[m_pr]
            p_sc = pr_sc[m_pr]

            # IoU match
            gm, pm, pred_indices = match_iou(
                g_ids, g_RT, g_sz, p_ids, p_RT, p_sz, p_sc, self.iou_thres,
                self.iou_pose_thres, self.synset_names)

            # sort predictions by score
            p_sc = p_sc[pred_indices]
            p_ids = p_ids[pred_indices]
            p_RT = p_RT[pred_indices]
            p_sz = p_sz[pred_indices]

            self.data[cid]['iou']['pm'].append(pm)
            self.data[cid]['iou']['scores'].append(p_sc)
            self.data[cid]['iou']['gm'].append(gm)

            # pose filtering
            if self.use_matches_for_pose:
                ti = self.iou_thres.index(self.iou_pose_thres)
                valid = pm[ti] >= 0
                p_ids, p_RT, p_sc = p_ids[valid], p_RT[valid], p_sc[valid]

                valid = gm[ti] >= 0
                g_ids, g_RT = g_ids[valid], g_RT[valid]

            # pose match
            gm_p, pm_p = match_rt(
                p_ids, p_RT, g_ids, g_RT, self.degree_thres, self.shift_thres,
                self.synset_names)

            self.data[cid]['pose']['pm'].append(pm_p)
            self.data[cid]['pose']['scores'].append(p_sc)
            self.data[cid]['pose']['gm'].append(gm_p)

    def compute_metrics(self):
        """Compute mAP for IoU and pose metrics."""

        # aggregate
        for cid in range(1, len(self.classes)):
            # iou
            pm = (np.concatenate(self.data[cid]['iou']['pm'], axis=1)
                  if self.data[cid]['iou']['pm']
                  else np.empty((len(self.iou_thres), 0)))
            sc = (np.concatenate(self.data[cid]['iou']['scores'], axis=0)
                  if self.data[cid]['iou']['scores']
                  else np.array([]))
            gm = (np.concatenate(self.data[cid]['iou']['gm'], axis=1)
                  if self.data[cid]['iou']['gm']
                  else np.empty((len(self.iou_thres), 0)))
            for t in range(len(self.iou_thres)):
                self.iou_ap[cid, t], self.iou_acc[cid, t] = \
                    compute_ap_acc(pm[t], sc, gm[t])

            # pose
            pm_p = (np.concatenate(self.data[cid]['pose']['pm'], axis=2)
                    if self.data[cid]['pose']['pm']
                    else np.empty((len(self.degree_thres)+1,
                                   len(self.shift_thres)+1, 0)))
            sc_p = (np.concatenate(self.data[cid]['pose']['scores'], axis=0)
                    if self.data[cid]['pose']['scores']
                    else np.array([]))
            gm_p = (np.concatenate(self.data[cid]['pose']['gm'], axis=2)
                    if self.data[cid]['pose']['gm']
                    else np.empty((len(self.degree_thres)+1,
                                   len(self.shift_thres)+1, 0)))
            for di in range(len(self.degree_thres)+1):
                for si in range(len(self.shift_thres)+1):
                    self.pose_ap[cid, di, si], self.pose_acc[cid, di, si] \
                        = compute_ap_acc(pm_p[di, si], sc_p, gm_p[di, si])

        # global
        self.iou_ap[0] = np.nanmean(self.iou_ap[1:], axis=0)
        self.iou_acc[0] = np.nanmean(self.iou_acc[1:], axis=0)
        self.pose_ap[0] = np.nanmean(self.pose_ap[1:], axis=0)
        self.pose_acc[0] = np.nanmean(self.pose_acc[1:], axis=0)

    def get_result(self):
        """Return the computed metrics."""
        res = self.print_metrics()
        # return results as 5 decimal place floats
        res = {k: round(float(v), 5) for k, v in res.items()}
        return res

    def print_metrics(self):
        # indices
        i25 = self.iou_thres.index(0.25) if 0.25 in self.iou_thres else 0
        i50 = self.iou_thres.index(0.5) if 0.5 in self.iou_thres else 0
        i75 = self.iou_thres.index(0.75) if 0.75 in self.iou_thres else 0
        d5 = self.degree_thres.index(5)
        d10 = self.degree_thres.index(10)
        s2 = self.shift_thres.index(2)
        s5 = self.shift_thres.index(5)

        header = f"{'Class':>20} "\
                 f"{'3DIoU@25':>9} {'3DIoU@50':>9} {'3DIoU@75':>9} " \
                 f"{'5°2cm':>9} {'5°5cm':>9} "\
                 f"{'10°2cm':>9} {'10°5cm':>9} " \
                 f"{'Acc@25':>9} {'Acc@50':>9} {'Acc@75':>9} " \
                 f"{'Acc5°2cm':>9} {'Acc5°5cm':>9} " \
                 f"{'Acc10°2cm':>9} {'Acc10°5cm':>9}"

        print("-" * len(header))
        print(header)
        print("-" * len(header))

        def _print(cid):
            row = [
                f"{self.classes[cid]:>20}",
                f"{self.iou_ap[cid, i25]:.3f}" if i25 else "N/A",
                f"{self.iou_ap[cid, i50]:.3f}" if i50 else "N/A",
                f"{self.iou_ap[cid, i75]:.3f}" if i75 else "N/A",
                f"{self.pose_ap[cid, d5, s2]:.3f}",
                f"{self.pose_ap[cid, d5, s5]:.3f}",
                f"{self.pose_ap[cid, d10, s2]:.3f}",
                f"{self.pose_ap[cid, d10, s5]:.3f}",
                f"{self.iou_acc[cid, i25]:.3f}" if i25 else "N/A",
                f"{self.iou_acc[cid, i50]:.3f}" if i50 else "N/A",
                f"{self.iou_acc[cid, i75]:.3f}" if i75 else "N/A",
                f"{self.pose_acc[cid, d5, s2]:.3f}",
                f"{self.pose_acc[cid, d5, s5]:.3f}",
                f"{self.pose_acc[cid, d10, s2]:.3f}",
                f"{self.pose_acc[cid, d10, s5]:.3f}"
            ]
            print("".join(f"{col:>10}" for col in row))

        for cid in range(len(self.classes)):
            _print(cid)
        print("-" * len(header))

        mean_results = {
            "3DIoU@25": self.iou_ap[0, i25] if i25 else None,
            "3DIoU@50": self.iou_ap[0, i50] if i50 else None,
            "3DIoU@75": self.iou_ap[0, i75] if i75 else None,
            "5°2cm": self.pose_ap[0, d5, s2],
            "5°5cm": self.pose_ap[0, d5, s5],
            "10°2cm": self.pose_ap[0, d10, s2],
            "10°5cm": self.pose_ap[0, d10, s5],
            "Acc@25": self.iou_acc[0, i25] if i25 else None,
            "Acc@50": self.iou_acc[0, i50] if i50 else None,
            "Acc@75": self.iou_acc[0, i75] if i75 else None,
            "Acc5°2cm": self.pose_acc[0, d5, s2],
            "Acc5°5cm": self.pose_acc[0, d5, s5],
            "Acc10°2cm": self.pose_acc[0, d10, s2],
            "Acc10°5cm": self.pose_acc[0, d10, s5]
        }
        return mean_results
