import sys
import importlib.util
from pathlib import Path

torch_spec = importlib.util.find_spec("torch")
assert torch_spec is not None and torch_spec.origin is not None, "Could not find torch"
torch_dir = Path(torch_spec.origin).parent

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

file_path = torch_dir / "version.py"
module = import_module_from_path("version", file_path)

if sys.argv[1] == "torch":
    print(module.__version__.split("+")[0])
elif sys.argv[1] == "cuda":
    print(module.cuda)
