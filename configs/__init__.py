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
        self.contrast_fraction = 0.5
        self.loss_type = "l2"
        self.experiment_name = ""
        self.debug = False
        self.viewer = False
        self.save_volume = False
        self.interpolation_start = 9_000
        self.interp_sigma_scale = 0.7
        self.interp_sigma_v = 0.35
        self.redundancy_threshold = 0.01   # relative to max activated density
        self.redundancy_cap = 0.05         # max fraction of cells removed per step
        self.rays_per_batch = 2_000_000
        super().__init__(parser, "Setting Pipeline parameters")


class ModelParams(ParamGroup):

    def __init__(self, parser):
        self.init_points = 32_000
        self.final_points = 128_000
        self.activation_scale = 1.0
        self.init_scale = 1.05
        self.init_type = "random"
        self.device = "cuda"
        super().__init__(parser, "Setting Model parameters")


class OptimizationParams(ParamGroup):

    def __init__(self, parser):
        self.points_lr_init = 2e-4
        self.points_lr_final = 5e-6
        self.density_lr_init = 5e-2
        self.density_lr_final = 1e-3
        self.freeze_points = 9_500
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
        super().__init__(parser, "Setting Optimization parameters")


class DatasetParams(ParamGroup):

    def __init__(self, parser):
        self.dataset = "r2_gaussian"
        self.data_path = "/mnt/hdd/r2_data/synthetic_dataset/cone_ntrain_75_angle_360/0_chest_cone"
        self.scene = "sphere"
        self.num_angles = 180
        self.detector_size = 128
        super().__init__(parser, "Setting Dataset parameters")
