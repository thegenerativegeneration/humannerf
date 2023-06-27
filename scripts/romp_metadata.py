import argparse
import os
import sys

import numpy as np

# cam (1, 3)
# global_orient (1, 3)
# body_pose (1, 69)
# smpl_betas (1, 10)
# smpl_thetas (1, 72)
# center_preds (1, 2)
# center_confs (1, 1)
# cam_trans (1, 3)
# verts (1, 6890, 3)
# joints (1, 71, 3)
# pj2d_org (1, 71, 2)


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='./video_results.npz')
parser.add_argument('-o', '--output', type=str, default='./metadata.json')

args = parser.parse_args(sys.argv[1:])

npz_path = args.path
if os.path.exists(npz_path):
    print('find video_results.npz in ', npz_path)
else:
    print(npz_path, ' not found')

FOV = 60
WIDTH = 1920
HEIGHT = 1080
focal_length_w = WIDTH/2. * 1./np.tan(np.radians(FOV / 2.))
focal_length_h = HEIGHT/2. * 1./np.tan(np.radians(FOV / 2.))

data = np.load(npz_path, allow_pickle=True) 
print(f"Loaded keys: {data.files}")
data_dict = data['results'][()]
# print(data['results']())
pose_output = []
n_frames = len(data_dict.keys())
print('n_frames ', n_frames)
out_metadata = {}
for i, (k, v) in enumerate(data_dict.items()):

    frame_name = k.split('.')[0]

    out_metadata[frame_name] = {
        "poses": v['smpl_thetas'][0].tolist(),
        "betas": v['smpl_betas'][0].tolist(),
        # TODO: there is something wrong with the camera parameters. Leads to pretty bad results.
        "cam_intrinsics": [
            [focal_length_w, 0.0,WIDTH / 2],    # focal length is 443.4, as FOV is 60 degrees
            [0.0, focal_length_h, HEIGHT / 2],   # 960 is half of 1920, 540 is half of 1080
            [0.0, 0.0, 1.0]
        ],
        "cam_extrinsics": np.eye(4).tolist(), # because ROMP works like this, see https://github.com/Arthur151/ROMP/issues/421
    }


with open(args.output, 'w') as f:
    import json
    json.dump(out_metadata, f, indent=4)
    print(f"Saved to {args.output}")