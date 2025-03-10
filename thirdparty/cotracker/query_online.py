# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor

# Unfortunately MPS acceleration does not support all the features we require,
# but we may be able to enable it in the future

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if __name__ == "__main__":
    print(DEFAULT_DEVICE)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/queries.mp4",
        help="path to a video",
    )

    parser.add_argument(
        "--query_path",
        default="./assets/queries.npy",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker2_online")
    model = model.to(DEFAULT_DEVICE)

    queries = np.load(args.query_path)
    queries = (torch.from_numpy(queries)[None]).to(DEFAULT_DEVICE)

    window_frames = []

    def _process_step(window_frames, is_first_step, queries):
        video_chunk = (
            torch.tensor(np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE)
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries,
        )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(
            args.video_path,
            plugin="FFMPEG",
        )
    ):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                queries=queries,
            )
            is_first_step = False
        window_frames.append(frame)
    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        queries=queries,
    )

    print(model.step)
    print("Tracks are computed")

    # save a video with predicted tracks
    seq_name = os.path.splitext(args.video_path.split("/")[-1])[0]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=0,
        filename=seq_name,
    )
