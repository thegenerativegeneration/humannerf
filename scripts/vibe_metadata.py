import argparse
import os

import numpy as np
import joblib


def get_camera_parameters(pred_cam, bbox):
    FOCAL_LENGTH = 5000.
    CROP_SIZE = 224

    bbox_cx, bbox_cy, bbox_w, bbox_h = bbox
    assert bbox_w == bbox_h

    bbox_size = bbox_w
    bbox_x = bbox_cx - bbox_w / 2.
    bbox_y = bbox_cy - bbox_h / 2.

    scale = bbox_size / CROP_SIZE

    cam_intrinsics = np.eye(3)
    cam_intrinsics[0, 0] = FOCAL_LENGTH * scale
    cam_intrinsics[1, 1] = FOCAL_LENGTH * scale
    cam_intrinsics[0, 2] = bbox_size / 2. + bbox_x 
    cam_intrinsics[1, 2] = bbox_size / 2. + bbox_y

    cam_s, cam_tx, cam_ty = pred_cam
    trans = [cam_tx, cam_ty, 2*FOCAL_LENGTH/(CROP_SIZE*cam_s + 1e-9)]

    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, 3] = trans

    return cam_intrinsics, cam_extrinsics

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='path to the pkl file')
parser.add_argument('-o', '--output', type=str, default='./metadata.json')

args = parser.parse_args()

pkl_path = args.path
if os.path.exists(pkl_path):
    print('find video_results.npz in ', pkl_path)
else:
    print(pkl_path, ' not found')



data = joblib.load(pkl_path)[1]  # the dict key is 1, for whatever reason
pose_output = []
n_frames = len(data["frame_ids"])
out_metadata = {}
for i in range(n_frames):
    frame_id = data["frame_ids"][i]
    bboxes = data["bboxes"][i]
    betas = data["betas"][i]
    poses = data["pose"][i]
    pred_cam = data["pred_cam"][i]

    cam_intrinsics, cam_extrinsics = get_camera_parameters(pred_cam, bboxes)

    frame_name = f"{frame_id:08d}"

    out_metadata[frame_name] = {
        "poses": poses.tolist(),
        "betas": betas.tolist(),
        "cam_intrinsics": cam_intrinsics.tolist(),
        "cam_extrinsics": cam_extrinsics.tolist(),
    }

    print(out_metadata[frame_name]["cam_intrinsics"])
    print(out_metadata[frame_name]["cam_extrinsics"])


with open(args.output, 'w') as f:
    import json
    json.dump(out_metadata, f, indent=4)
    print(f"Saved to {args.output}")