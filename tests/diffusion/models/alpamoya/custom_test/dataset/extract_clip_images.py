import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
OFFLINE_DIR = REPO_ROOT / "tests" / "diffusion" / "models" / "alpamoya" / "custom_test" / "offline"

for path in (REPO_ROOT, OFFLINE_DIR):
    path_str = str(path)
    if path.is_dir() and path_str not in sys.path:
        sys.path.insert(0, path_str)

import common as ct
from alpamayo1_5 import helper


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Alpamayo dataset images for one clip.")
    parser.add_argument("--clip-id", type=str, default=ct.CLIP_ID, help="Clip ID to extract")
    parser.add_argument("--t0-us", type=int, default=ct.T0_US, help="Timestamp passed to dataset loader")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory where extracted PNG files are written",
    )
    args = parser.parse_args()

    data = ct.load_clip_data(clip_id=args.clip_id, t0_us=int(args.t0_us))
    image_frames = data["image_frames"]
    camera_indices = data["camera_indices"]

    clip_dir = args.output_dir / str(args.clip_id)
    clip_dir.mkdir(parents=True, exist_ok=True)

    num_cameras = int(image_frames.shape[0])
    num_frames = int(image_frames.shape[1])
    print(f"clip_id={args.clip_id} t0_us={int(args.t0_us)} num_cameras={num_cameras} num_frames={num_frames}")
    print(f"output_dir={clip_dir}")

    for camera_pos, cam_id_tensor in enumerate(camera_indices):
        cam_id = int(cam_id_tensor.item())
        cam_name = helper.CAMERA_DISPLAY_NAMES.get(cam_id, f"camera_{cam_id}")
        camera_slug = cam_name.lower().replace(" ", "_")
        camera_dir = clip_dir / f"{camera_pos:02d}_{camera_slug}"
        camera_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(num_frames):
            image = ct.tensor_to_pil(image_frames[camera_pos, frame_idx])
            output_path = camera_dir / f"frame_{frame_idx:03d}.png"
            image.save(output_path)

        print(
            f"saved camera_index={camera_pos} camera_id={cam_id} "
            f"camera_name={cam_name} frames={num_frames} dir={camera_dir}"
        )


if __name__ == "__main__":
    main()
