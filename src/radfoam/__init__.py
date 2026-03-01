import os
import warnings
import ctypes

import torch

torch_version_at_compile = "2.5.1"
torch_version_at_runtime = torch.__version__.split("+")[0]

if torch_version_at_compile != torch_version_at_runtime:
    warnings.warn(
        f"RadFoam was compiled with torch version {torch_version_at_compile}, but "
        f"the current torch version is {torch_version_at_runtime}. This might lead to "
        "unexpected behavior or crashes."
    )

if True:
    try:
        import glfw
        glfw_path = os.path.split(glfw.__file__)[0]
        libglfw_path = os.path.join(glfw_path, "x11", "libglfw.so")
        libdl = ctypes.CDLL("libdl.so.2")
        RTLD_NOW = 0x00002
        RTLD_GLOBAL = 0x00100
        handle = libdl.dlopen(libglfw_path.encode("utf-8"), RTLD_NOW | RTLD_GLOBAL)
        if handle is None:
            raise ImportError("failed to load libglfw.so")
    except (ImportError, OSError):
        warnings.warn(
            "GLFW not available — the viewer will not work. "
            "Install glfw (`pip install glfw`) if you need the viewer."
        )

from .torch_bindings import *
