import argparse
import torch
import radfoam


def main():
    parser = argparse.ArgumentParser(description="View a CT reconstruction")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model.pt file"
    )
    parser.add_argument(
        "--camera_distance", type=float, default=60.0,
        help="Initial camera distance from origin",
    )
    parser.add_argument(
        "--up", type=str, default="Y", choices=["X", "Y", "Z"],
        help="Which axis is up (default: Y)",
    )
    args = parser.parse_args()

    device = torch.device("cuda")

    scene_data = torch.load(args.model, map_location=device)
    xyz = scene_data["xyz"].to(device).float().contiguous()
    density = scene_data["density"].to(device).float().contiguous()
    adjacency = scene_data["adjacency"].to(device).to(torch.uint32).contiguous()
    adjacency_offsets = scene_data["adjacency_offsets"].to(device).to(torch.uint32).contiguous()

    aabb_tree = radfoam.build_aabb_tree(xyz)
    pipeline = radfoam.create_ct_pipeline()

    camera_pos = torch.tensor([0.0, 0.0, args.camera_distance], dtype=torch.float32)
    camera_forward = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    camera_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    def callback(viewer):
        viewer.update_scene(xyz, density, adjacency, adjacency_offsets, aabb_tree)

    orbit_target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    radfoam.run_with_viewer(
        pipeline,
        callback,
        camera_pos=camera_pos,
        camera_forward=camera_forward,
        camera_up=camera_up,
        orbit_target=orbit_target,
    )


if __name__ == "__main__":
    main()
