"""Open-vocabulary text-guided segmentation via CLIP-aligned cell features.

Uses a CLIP-compatible feature backbone (MedCLIP or standard OpenCLIP) to encode
a text prompt and find Voronoi cells whose features are most similar.

Requires:
    pip install open-clip-torch        # for standard OpenCLIP backbones
    pip install git+https://github.com/microsoft/MedCLIP   # for MedCLIP

Usage
-----
    micromamba run -n radfoam python demos/text_segment.py \\
        --config output/<run>/config.yaml \\
        --features output/<run>/features_<backbone>.npz \\
        --text "spine" \\
        [--backbone openai/clip-vit-base-patch32] \\
        [--topk 0.05] \\
        [--out output/<run>/text_segment.png]

This demo is the open-vocab analogue to click_segment.py, and is the key
differentiator vs volumetric 3DGS: unambiguous spatial assignment makes it
possible to attach CLIP-aligned features to cells directly, then query them
with natural language.
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vis_foam import load_gt_volume, load_density_field
from radfoam_model.features import load_cell_features


def load_clip_text_encoder(backbone="openai/clip-vit-base-patch32"):
    """Load a CLIP text encoder.

    Args:
        backbone: model identifier for open-clip (e.g. "openai/clip-vit-base-patch32")
                  or "medclip" for MedCLIP.

    Returns:
        (encode_text, tokenize) callables
    """
    if backbone == "medclip":
        try:
            from medclip import MedCLIPModel, MedCLIPProcessor
        except ImportError:
            raise ImportError(
                "MedCLIP is not installed.\n"
                "Run: pip install git+https://github.com/microsoft/MedCLIP"
            )
        raise NotImplementedError("MedCLIP text encoder not yet wired up.")

    try:
        import open_clip
    except ImportError:
        raise ImportError(
            "open-clip-torch is not installed.\n"
            "Run: pip install open-clip-torch\n"
            "Then retry."
        )

    # Parse "org/model_name" format
    parts = backbone.split("/", 1)
    if len(parts) == 2 and parts[0] == "openai":
        model_name = parts[1]
        pretrained = "openai"
    elif len(parts) == 2:
        model_name = parts[0]
        pretrained = parts[1]
    else:
        model_name = backbone
        pretrained = "openai"

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


def encode_text_prompt(model, tokenizer, text, device="cuda"):
    """Encode a text string with CLIP and return a unit-normalised (F,) float32 tensor."""
    import torch
    tokens = tokenizer([text]).to(device)
    model = model.to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
        feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat.squeeze(0)


def cosine_similarity_batch(query_feat, cell_feats):
    import torch
    q = torch.nn.functional.normalize(query_feat.float().unsqueeze(0), dim=1)
    c = torch.nn.functional.normalize(cell_feats.float(), dim=1)
    return (c @ q.T).squeeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--text",     required=True, nargs="+",
                        help="Text prompt(s) (multiple = averaged)")
    parser.add_argument("--backbone", default="openai/ViT-B-32",
                        help="OpenCLIP model identifier (default: openai/ViT-B-32)")
    parser.add_argument("--topk",     type=float, default=0.05)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--out",      default=None)
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    run_dir = os.path.dirname(config_path)
    out_path = args.out or os.path.join(
        run_dir, f"text_segment_{'_'.join(args.text[:2])}.png"
    )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_path = cfg.get("data_path", "")
    dataset   = cfg.get("dataset", "r2_gaussian")

    import torch

    # Load model checkpoint
    field  = load_density_field(os.path.join(run_dir, "model.pt"), device=args.device)
    points = field["points"]
    N = points.shape[0]

    # Load cell features
    cell_feats_np, meta = load_cell_features(args.features)
    print(f"Cell features: {cell_feats_np.shape}  backbone={meta.get('backbone', '?')}")
    cell_feats = torch.from_numpy(cell_feats_np.astype(np.float32)).to(args.device)

    # Encode text prompts
    model, tokenizer = load_clip_text_encoder(args.backbone)
    text_feats = []
    for t in args.text:
        feat = encode_text_prompt(model, tokenizer, t, device=args.device)
        text_feats.append(feat)
    query_feat = torch.stack(text_feats).mean(0)

    sim = cosine_similarity_batch(query_feat, cell_feats)
    N_top = max(1, int(args.topk * N))
    thresh_val = float(sim.topk(N_top).values[-1].item())
    sim_np = sim.cpu().numpy()
    sim_norm = np.clip((sim_np - thresh_val) / (1.0 - thresh_val + 1e-8), 0, 1)
    print(f"Similarity: min={sim_np.min():.3f}  max={sim_np.max():.3f}  "
          f"threshold={thresh_val:.3f}")

    # Visualise (reuse click_segment projection logic)
    gt_volume = load_gt_volume(data_path, dataset)
    if gt_volume is None:
        gt_volume = np.zeros((128, 128, 128), dtype=np.float32)
    R = gt_volume.shape[0]
    points_np = points.cpu().numpy()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from click_segment import project_sim_to_slices, make_figure

    def world_to_vox_f(w):
        return (np.array(w) + 1.0) / 2.0 * (R - 1)

    center = np.zeros(3, dtype=np.float32)
    gt_slices = {
        "axial":    np.take(gt_volume, R // 2, axis=0),
        "coronal":  np.take(gt_volume, R // 2, axis=1),
        "sagittal": np.take(gt_volume, R // 2, axis=2),
    }
    sim_slices = project_sim_to_slices(sim_norm, points_np, gt_volume)

    import matplotlib.pyplot as plt
    fig = make_figure(sim_slices, gt_slices, center, world_to_vox_f(center))
    fig.suptitle(f"Text query: \"{' '.join(args.text)}\"", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
