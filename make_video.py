#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def make_video_opencv(image_dir, output_path, fps):
    try:
        import cv2
    except Exception:
        return False, "opencv not available"

    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        return False, "no images found"

    first = cv2.imread(os.path.join(image_dir, images[0]))
    if first is None:
        return False, "failed to read first image"

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for fname in images:
        img = cv2.imread(os.path.join(image_dir, fname))
        if img is None:
            print(f"Warning: failed to read {fname}, skipping")
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    return True, None

def make_video_ffmpeg(image_dir, output_path, fps):
    # Prefer sequence pattern frame_0000.png; fallback to glob via ffmpeg if not present.
    seq_path = os.path.join(image_dir, "frame_%04d.png")
    if any(f.startswith("frame_") and f.endswith(".png") for f in os.listdir(image_dir)):
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-start_number", "0",
            "-i", seq_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]
    else:
        glob_pattern = os.path.join(image_dir, "*.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", glob_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0, proc.stderr.decode().strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", "-i", default="/home/ved/Ved/Project_1/dataset_visual_v2/images")
    p.add_argument("--out", "-o", default="/home/ved/Ved/Project_1/dataset_visual_v2/video.mp4")
    p.add_argument("--fps", "-r", type=int, default=30)
    args = p.parse_args()

    if not os.path.isdir(args.images):
        print("Image directory not found:", args.images)
        sys.exit(1)

    ok, err = make_video_opencv(args.images, args.out, args.fps)
    if ok:
        print("Video written:", args.out)
        return
    # fallback to ffmpeg
    ok, err = make_video_ffmpeg(args.images, args.out, args.fps)
    if ok:
        print("Video written with ffmpeg:", args.out)
        return

    print("Failed to create video. Reason:", err)
    sys.exit(1)

if __name__ == "__main__":
    main()
