import configargparse
import os
from argparse import Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(
        self, parser: configargparse.ArgParser, name: str, fill_none=False
    ):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            t = type(value)
            value = value if not fill_none else None
            if t == bool:
                group.add_argument(
                    "--" + key, default=value, action="store_true"
                )
            elif t == list:
                group.add_argument(
                    "--" + key,
                    nargs="+",
                    type=type(value[0]),
                    default=value,
                    help=f"List of {type(value[0]).__name__}",
                )
            else:
                group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class PipelineParams(ParamGroup):

    def __init__(self, parser):
        self.iterations = 10_000
        self.densify_from = 1_000
        self.densify_until = 6_000
        self.densify_factor = 1.15
        self.gradient_fraction = 0.4
        self.idw_fraction = 0.3
        self.entropy_fraction = 0.3
        self.entropy_bins = 5
        self.contrast_alpha = 0.0
        self.loss_type = "l2"
        self.experiment_name = ""
        self.debug = False
        self.viewer = False
        self.save_volume = False
        self.interpolation_start = 9_000
        self.interp_sigma_scale = 0.7
        self.interp_sigma_v = 0.35
        self.per_cell_sigma = False
        self.per_neighbor_sigma = False
        self.redundancy_threshold = 0.01   # relative to max activated density
        self.interp_ramp = False
        self.redundancy_cap = 0.05         # max fraction of cells removed per step (constant)
        self.redundancy_cap_init = 0.0     # adaptive cap schedule: start value (0 = use redundancy_cap)
        self.redundancy_cap_final = 0.0    # adaptive cap schedule: end value
        self.prune_variance_criterion = False  # use neighborhood variance instead of IDW error
        self.prune_hops = 1                # k-hop neighborhood for variance pruning
        self.rays_per_batch = 2_000_000
        self.bf_start = -1            # iteration to start (-1 = disabled)
        self.bf_until = 6000          # iteration to stop
        self.bf_period = 10           # apply every N steps
        self.bf_sigma_init = 2.0      # initial spatial sigma (x per-cell radius)
        self.bf_sigma_final = 0.3     # final spatial sigma
        self.bf_sigma_v_init = 10.0   # initial value sigma (high = Gaussian blur)
        self.bf_sigma_v_final = 0.1   # final value sigma (low = bilateral)
        self.targeted_fraction = 0.0  # 0 = disabled, 0.2 = 20% targeted rays
        self.targeted_start = -1      # iteration to start (-1 = same as densify_from)
        self.high_error_fraction = 0.0  # 0 = disabled, 0.2 = 20% high-error rays
        self.high_error_power = 1.0     # power scaling on error weights (1=linear, 2=quadratic)
        self.high_error_start = -1      # iteration to start (-1 = same as densify_from)
        self.log_percent = 5          # log metrics every N% of iterations
        self.diag_percent = 10        # log diagnostics/slices every N% of iterations
        super().__init__(parser, "Setting Pipeline parameters")


class ModelParams(ParamGroup):

    def __init__(self, parser):
        self.init_points = 32_000
        self.final_points = 128_000
        self.activation_scale = 1.0
        self.init_scale = 1.05
        self.init_type = "random"
        self.init_density = 0.0
        self.device = "cuda"
        self.init_points_file = ""
        self.init_volume_path = ""  # path to .npy volume for density init (e.g. FDK)
        self.frozen_points_file = ""      # path to model.pt whose xyz+density start frozen
        self.frozen_freeze_density = True # also freeze density of old points until unfreeze
        super().__init__(parser, "Setting Model parameters")


class OptimizationParams(ParamGroup):

    def __init__(self, parser):
        self.points_lr_init = 2e-4
        self.points_lr_final = 5e-6
        self.density_lr_init = 5e-2
        self.density_lr_final = 1e-3
        self.freeze_points = 9_500
        self.frozen_unfreeze_step = -1    # iter to unfreeze loaded frozen points (-1 = never)
        self.tv_weight = 1e-4
        self.tv_start = 5_000
        self.tv_epsilon = 1e-4
        self.tv_area_weighted = False
        self.tv_border = False
        self.gradient_start = -1
        self.gradient_lr_init = 1e-2
        self.gradient_lr_final = 1e-3
        self.gradient_warmup = 500
        self.gradient_max_slope = 5.0
        self.gradient_freeze_points = 500
        self.tv_anneal = False
        self.tv_on_raw = False
        self.voxel_var_weight = 0.0
        self.voxel_var_weight_final = -1.0  # weight at interpolation_start (-1 = hold voxel_var_weight)
        self.voxel_var_resolution = 32
        self.voxel_var_sigma_v = 0.2       # kept for backward compat; overridden by var_sigma_v_*
        self.voxel_var_start = 0
        self.neighbor_var_weight = 0.0     # graph neighbor variance regularization weight
        self.neighbor_var_weight_final = -1.0  # weight at interpolation_start (-1 = hold neighbor_var_weight)
        self.neighbor_var_hops = 1         # k-hop neighborhood depth
        self.neighbor_var_start = 0
        self.var_sigma_v_init = 0.2        # bilateral sigma at start (large = plain smoothing)
        self.var_sigma_v_final = 0.2       # bilateral sigma at end (small = edge-preserving)
        self.density_grad_clip = 1.0
        self.ref_volume_path = ""           # path to .npy or .pt reference volume (empty = off)
        self.ref_volume_weight = 0.0        # L2 loss weight (0 = disabled)
        self.ref_volume_weight_final = -1.0 # weight at interpolation_start (-1 = hold weight)
        self.ref_volume_start = 0           # activation step
        self.ref_volume_until = -1          # deactivation step (-1 = never)
        self.ref_volume_resolution = 64     # voxel grid resolution for loss computation
        self.ref_volume_blur_sigma = 2.0    # Gaussian blur applied to reference (in source voxels)
        self.ref_volume_edge_mask = False   # weight loss by inverse gradient magnitude
        self.ref_volume_edge_alpha = 10.0   # edge mask sensitivity: 1/(1+alpha*|∇ref|)
        self.gaussian_start = -1
        self.freeze_base_at_gaussian = False
        self.joint_finetune_start = -1
        self.peak_lr_init = 1e-2
        self.peak_lr_final = 1e-3
        self.offset_lr_init = 1e-3
        self.offset_lr_final = 1e-4
        self.cov_lr_init = 1e-2
        self.cov_lr_final = 1e-3
        super().__init__(parser, "Setting Optimization parameters")


class DatasetParams(ParamGroup):

    def __init__(self, parser):
        self.dataset = "r2_gaussian"
        self.data_path = "r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
        self.scene = "sphere"
        self.num_angles = 180
        self.detector_size = 128
        self.sample_index = 0
        self.mode = 1
        self.split_override = ""
        super().__init__(parser, "Setting Dataset parameters")
