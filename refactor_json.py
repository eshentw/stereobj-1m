import json
import os
import argparse
from tqdm import tqdm


def parser():
    parser = argparse.ArgumentParser(description='Merge JSON prediction files of all object classes')
    parser.add_argument('--gt_json', required=True, help='Input directory containing JSON files')
    parser.add_argument('--output_dir', required=True, help='Output directory for merged JSON file')
    parser.add_argument('--split', default='val', help='Dataset split [default: val]')
    return parser

def refactor_json(args):
    output_dir = args.output_dir
    split = args.split
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    centrifuge_seq_id = [
            "biolab_scene_1_07162020_14",
            "biolab_scene_1_07182020_5",
            "biolab_scene_3_08022020_4",
            "biolab_scene_8_08172020_4",
            ]

    # biolab_scene_1_07162020_14: {
    #                       frame_id: {
    #                                   cls_type1: pose_pred
    #                                   cls_type2: pose_pred
    #                                }
    #                 }
    #       }

    with open(args.gt_json, 'r') as f:
        gt_dict = json.load(f)
    gt_dict = gt_dict['pred']

    refactored_dict = {'split': split}
    
    for cls_type in gt_dict:
        for seq_id in gt_dict[cls_type]:
            if seq_id in centrifuge_seq_id: 
                continue
            if seq_id not in refactored_dict:
                refactored_dict[seq_id] = {}
            for frame_id in tqdm(gt_dict[cls_type][seq_id], desc=f"Processing {seq_id}"):
                if frame_id not in refactored_dict[seq_id]:
                    refactored_dict[seq_id][frame_id] = {}
                pose_pred = gt_dict[cls_type][seq_id][frame_id]
                refactored_dict[seq_id][frame_id][cls_type] = pose_pred


    save_filename = os.path.join(output_dir, 'refactor_gt_pred.json')
    with open(save_filename, 'w') as f:
        json.dump(refactored_dict, f, indent=1)
    print(f'Refactor JSON saved to: {save_filename}')
    
if __name__ == "__main__":
    args = parser().parse_args()
    refactor_json(args)