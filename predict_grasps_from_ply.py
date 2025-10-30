#!/usr/bin/env python3
# Standalone script to predict grasps on a PLY point cloud file
import argparse
import numpy as np
import torch
import open3d as o3d

from m2t2.dataset import collate
from m2t2.dataset_utils import denormalize_rgb, sample_points
from m2t2.meshcat_utils import (
    create_visualizer, make_frame, visualize_grasp, visualize_pointcloud
)
from m2t2.m2t2 import M2T2
from m2t2.plot_utils import get_set_colors
from m2t2.train_utils import to_cpu, to_gpu


def load_ply_with_rgb(ply_path):
    """Load a PLY file and extract XYZ and RGB data using Open3D."""
    pcd = o3d.io.read_point_cloud(ply_path)

    # Extract XYZ coordinates
    xyz = np.asarray(pcd.points)

    # Extract RGB colors (Open3D returns colors in 0-1 range)
    if pcd.has_colors():
        rgb = np.asarray(pcd.colors)
    else:
        # Default white color if no RGB data
        rgb = np.ones_like(xyz)

    print(f"Loaded {xyz.shape[0]} points from {ply_path}")

    return torch.from_numpy(xyz).float(), torch.from_numpy(rgb).float()


def normalize_rgb(rgb):
    """Normalize RGB values."""
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return (rgb - mean) / std


def prepare_data_from_ply(ply_path, num_points=16384, world_coord=True):
    """Prepare data dictionary from PLY file in the format expected by M2T2."""
    xyz, rgb = load_ply_with_rgb(ply_path)

    # Apply spatial bounds to filter the point cloud
    # x in [0.0, 1.0], y in [-0.7, 0.7], z in [-0.2, 0.5]
    bounds_mask = (
        (xyz[:, 0] >= 0.0) & (xyz[:, 0] <= 1.0) &
        (xyz[:, 1] >= -0.3) & (xyz[:, 1] <= 0.3) &
        (xyz[:, 2] >= -0.2) & (xyz[:, 2] <= 0.5)
    )
    xyz = xyz[bounds_mask]
    rgb = rgb[bounds_mask]

    print(f"After applying bounds: {xyz.shape[0]} points remaining")
    print(f"  x: [0.0, 1.0], y: [-0.7, 0.7], z: [-0.2, 0.5]")

    # Normalize RGB
    rgb_normalized = normalize_rgb(rgb)

    # Center the point cloud
    xyz_centered = xyz - xyz.mean(dim=0)

    # Create camera pose (identity for world coordinates)
    cam_pose = torch.eye(4).float()

    # Prepare inputs (centered xyz + normalized rgb)
    inputs = torch.cat([xyz_centered, rgb_normalized], dim=1)

    # Create dummy segmentation (all points belong to the scene)
    seg = torch.ones(xyz.shape[0], dtype=torch.long)

    data = {
        'inputs': inputs,
        'points': xyz,
        'seg': seg,
        'cam_pose': cam_pose,
        'task': 'pick',  # For grasp prediction
        # Dummy object inputs (not used for pick task but required by model)
        'object_inputs': torch.rand(1024, 6),
        'ee_pose': torch.eye(4).float(),
        'bottom_center': torch.zeros(3),
        'object_center': torch.zeros(3)
    }

    return data, xyz, rgb


def predict_grasps(ply_path, checkpoint_path, num_points=16384, num_runs=5,
                   mask_thresh=0.4, world_coord=True):
    """Load PLY file and predict grasps."""
    print(f"Loading point cloud from {ply_path}...")
    data, xyz_original, rgb_original = prepare_data_from_ply(
        ply_path, num_points, world_coord
    )

    print(f"Loading model from {checkpoint_path}...")
    # Load model
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('config.yaml')

    model = M2T2.from_config(cfg.m2t2)
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt['model'])
    model = model.cuda().eval()

    print(f"Running inference with {num_runs} runs...")
    inputs, xyz, seg = data['inputs'], data['points'], data['seg']
    obj_inputs = data['object_inputs']

    outputs = {
        'grasps': [],
        'grasp_confidence': [],
        'grasp_contacts': []
    }

    # Run multiple forward passes
    for run_idx in range(num_runs):
        # Sample points
        pt_idx = sample_points(xyz, num_points)
        data['inputs'] = inputs[pt_idx]
        data['points'] = xyz[pt_idx]
        data['seg'] = seg[pt_idx]
        pt_idx = sample_points(obj_inputs, cfg.data.num_object_points)
        data['object_inputs'] = obj_inputs[pt_idx]

        # Prepare batch
        data_batch = collate([data])
        to_gpu(data_batch)

        # Inference
        with torch.no_grad():
            cfg.eval.mask_thresh = mask_thresh
            cfg.eval.object_thresh = mask_thresh
            model_outputs = model.infer(data_batch, cfg.eval)
        to_cpu(model_outputs)

        # Aggregate outputs
        for key in outputs:
            outputs[key].extend(model_outputs[key][0])

        print(f"  Run {run_idx+1}/{num_runs} complete")

    # Restore original data
    data['inputs'], data['points'], data['seg'] = inputs, xyz, seg
    data['object_inputs'] = obj_inputs

    return data, outputs, rgb_original


def visualize_results(data, outputs, rgb_original, world_coord=True):
    """Visualize the point cloud and predicted grasps."""
    print("\nVisualizing results...")
    print("Open http://127.0.0.1:7000/static/ in your browser to see the visualization")

    vis = create_visualizer()

    # Prepare RGB for visualization
    rgb_viz = denormalize_rgb(
        data['inputs'][:, 3:].T.unsqueeze(2)
    ).squeeze(2).T
    rgb_viz = (rgb_viz.numpy() * 255).astype('uint8')

    # Get XYZ
    xyz = data['points'].numpy()
    cam_pose = data['cam_pose'].double().numpy()

    # Create camera frame
    make_frame(vis, 'camera', T=cam_pose)

    # Transform to world coordinates if needed
    if not world_coord:
        xyz = xyz @ cam_pose[:3, :3].T + cam_pose[:3, 3]

    # Visualize point cloud
    visualize_pointcloud(vis, 'scene', xyz, rgb_viz, size=0.005)

    # Visualize grasps
    if data['task'] == 'pick':
        total_grasps = 0
        for i, (grasps, conf, contacts, color) in enumerate(zip(
            outputs['grasps'],
            outputs['grasp_confidence'],
            outputs['grasp_contacts'],
            get_set_colors()
        )):
            num_grasps = grasps.shape[0]
            total_grasps += num_grasps
            print(f"  Object {i:02d}: {num_grasps} grasps predicted")

            # Visualize contact points with confidence colors
            conf_np = conf.numpy()
            conf_colors = (np.stack([
                1 - conf_np, conf_np, np.zeros_like(conf_np)
            ], axis=1) * 255).astype('uint8')
            visualize_pointcloud(
                vis, f"object_{i:02d}/contacts",
                contacts.numpy(), conf_colors, size=0.01
            )

            # Visualize grasp poses
            grasps_np = grasps.numpy()
            if not world_coord:
                grasps_np = cam_pose @ grasps_np

            for j, grasp in enumerate(grasps_np):
                visualize_grasp(
                    vis, f"object_{i:02d}/grasps/{j:03d}",
                    grasp, color, linewidth=0.2
                )

        print(f"\nTotal grasps predicted: {total_grasps}")

    print("\nVisualization complete!")
    print("The visualization will remain open. Press Ctrl+C to exit.")


def main():
    parser = argparse.ArgumentParser(
        description='Predict grasps on a PLY point cloud file'
    )
    parser.add_argument(
        'ply_path',
        type=str,
        help='Path to the input PLY file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='m2t2.pth',
        help='Path to model checkpoint (default: m2t2.pth)'
    )
    parser.add_argument(
        '--num-points',
        type=int,
        default=16384,
        help='Number of points to sample (default: 16384)'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=5,
        help='Number of inference runs to aggregate (default: 5)'
    )
    parser.add_argument(
        '--mask-thresh',
        type=float,
        default=0.2,
        help='Confidence threshold for predicted grasps (default: 0.3)'
    )
    parser.add_argument(
        '--world-coord',
        action='store_true',
        default=True,
        help='Use world coordinates (default: True)'
    )

    args = parser.parse_args()

    # Check if meshcat server is running
    print("=" * 80)
    print("IMPORTANT: Make sure meshcat-server is running in a separate terminal!")
    print("  Run: meshcat-server")
    print("  Then open: http://127.0.0.1:7000/static/ in your browser")
    print("=" * 80)
    print()

    # Predict grasps
    data, outputs, rgb_original = predict_grasps(
        args.ply_path,
        args.checkpoint,
        num_points=args.num_points,
        num_runs=args.num_runs,
        mask_thresh=args.mask_thresh,
        world_coord=args.world_coord
    )

    # Visualize
    visualize_results(data, outputs, rgb_original, args.world_coord)

    # Keep the script running
    # try:
    #     import time
    #     while True:
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("\nExiting...")


if __name__ == '__main__':
    main()
