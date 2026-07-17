"""Regression test: train.py logging must handle non-numeric entries
in `thin_surface_diagnostics()` without crashing.

CH8 (relative chest) reached correct thin-surface activation, then crashed
inside the train.py diagnostics loop:

    for _k, _v in _ts_diag.items():
        writer.add_scalar(f"thin/{_k}", _v, i)

because the M5 relative-delta prototype added a categorical `delta_mode`
key (value "relative" / "absolute") to the diagnostics dict.  TensorBoard's
`SummaryWriter.add_scalar` rejects strings with:

    ValueError: could not convert string to float: 'relative'

This test guards three things:
  1. The diagnostics dict always carries a string-valued `delta_mode`.
  2. The fixed logging helper `_log_diag_kv` in train.py routes numeric
     entries to `add_scalar` and string entries to `add_text` -- never
     the other way around.
  3. The unchanged TensorBoard `add_scalar` raises ValueError on a string,
     so a future regression that removes the guard would be caught.

The test is CPU-only and runs without a CUDA device.  It imports the
helper from `train.py` directly so it verifies the real code path --
not a parallel copy.

Run:  micromamba run -n radfoam python test/test_thin_surface_diag_logging.py
"""

import os
import sys
import types
import tempfile
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# CPU radfoam stub (mirrors other thin-surface tests).
# ---------------------------------------------------------------------------
def _install_radfoam_stub():
    if "radfoam" in sys.modules:
        return
    mod = types.ModuleType("radfoam")
    mod.build_aabb_tree = lambda pts: None
    mod.farthest_neighbor = lambda *a, **k: (None, None)
    mod.nn = lambda *a, **k: None
    mod.TriangulationFailedError = type(
        "TriangulationFailedError", (Exception,), {})
    mod.Triangulation = None
    mod.BatchFetcher = lambda *a, **k: None
    mod.create_ct_pipeline = lambda: None
    sys.modules["radfoam"] = mod


_install_radfoam_stub()

from radfoam_model.scene import CTScene  # noqa: E402

# Import the helper from the real train.py module so this test verifies
# the actual code path -- not a parallel copy.  train.py is a script with
# an `if __name__ == "__main__":` guard, so importing is side-effect-free.
import train as _train_mod  # noqa: E402
assert hasattr(_train_mod, "_log_diag_kv"), (
    "train.py must define module-level `_log_diag_kv` (see CH8 fix)"
)
_log_diag_kv = _train_mod._log_diag_kv
# Static guard: train.py's train() must call `_log_diag_kv` -- not the
# raw `add_scalar` -- in the P0-F thin-surface diagnostics branch.
_train_src = open(os.path.join(
    os.path.dirname(__file__), "..", "train.py")).read()
assert "_log_diag_kv(writer" in _train_src, (
    "train.py P0-F branch must call _log_diag_kv (raw add_scalar would "
    "crash on the categorical delta_mode entry)"
)

torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Helpers (lightweight, copied from test_thin_surface_relative_delta.py --
# kept self-contained so a regression in either test file is isolated).
# ---------------------------------------------------------------------------
def _args(relative=False, rho=0.5):
    class A:
        pass
    a = A()
    a.points_lr_init = 2e-4
    a.points_lr_final = 5e-6
    a.density_lr_init = 5e-2
    a.density_lr_final = 1e-2
    a.freeze_points = 9500
    a.thin_surface_start = 0
    a.thin_surface_K = 4
    a.thin_surface_delta_weight = 1e-3
    a.thin_surface_height_weight = 5e-4
    a.thin_surface_gate_tau = 0.01
    a.thin_surface_lr_scale = 1.0
    a.thin_surface_delta_lr_scale = 1.0
    a.thin_surface_quat_lr_scale = 1.0
    a.thin_surface_sites_lr_scale = 1.0
    a.thin_surface_heights_lr_scale = 1.0
    a.thin_surface_delta_clip = 2.0
    a.thin_surface_grad_clip = 1.0
    a.thin_surface_relative_delta = relative
    a.thin_surface_delta_max_frac = rho
    return a


def _make_scene(n_points=8, device="cpu"):
    scene = object.__new__(CTScene)
    nn.Module.__init__(scene)
    scene.activation_scale = 1.0
    scene.device = torch.device(device)
    scene.num_init_points = n_points
    scene.num_final_points = n_points
    scene._thin_surface_active = False
    scene._thin_K = 4
    scene._thin_surface_gate_tau = 0.01
    scene.thin_surface_scheduler_args = None

    pts = (torch.rand(n_points, 3) - 0.5) * 1.0
    scene.primal_points = nn.Parameter(pts)
    scene.density = nn.Parameter(
        torch.linspace(0.2, 2.0, n_points).unsqueeze(-1))

    adj = []
    offsets = [0]
    for i in range(n_points):
        nbrs = [j for j in range(n_points) if j != i]
        adj.extend(nbrs)
        offsets.append(len(adj))
    scene.point_adjacency = (
        torch.tensor(adj, dtype=torch.int32).to(torch.uint32))
    scene.point_adjacency_offsets = (
        torch.tensor(offsets, dtype=torch.int32).to(torch.uint32))
    scene._cached_cell_radius = torch.ones(n_points)
    return scene


def _log_diag_dict(writer, diag, step):
    """Run the diagnostics dict through train.py's real `_log_diag_kv`
    helper (imported above) once per key.  Counts add_scalar/add_text
    invocations on writers that expose those counters; otherwise returns
    None so callers that only need a no-exception smoke test can ignore."""
    n_scalar = 0
    n_text = 0
    for k, v in diag.items():
        before_scalar = getattr(writer, "add_scalar_calls", None)
        before_text = getattr(writer, "add_text_calls", None)
        _log_diag_kv(writer, k, v, step)
        if before_scalar is not None:
            n_scalar += writer.add_scalar_calls - before_scalar
        if before_text is not None:
            n_text += writer.add_text_calls - before_text
    return {"n_scalar": n_scalar, "n_text": n_text}


# ---------------------------------------------------------------------------
# Tiny in-memory SummaryWriter stand-in that records every call.  Avoids
# touching disk and gives us precise control over what TensorBoard sees.
# ---------------------------------------------------------------------------
class _RecordingWriter:
    def __init__(self):
        self.scalars = []   # list[(tag, value, step)]
        self.texts = []     # list[(tag, value, step)]
        self.add_scalar_calls = 0
        self.add_text_calls = 0

    def add_scalar(self, tag, scalar_value, global_step=None, **kw):
        self.add_scalar_calls += 1
        # Mirror real SummaryWriter: ValueError on string (which it gets
        # via float()).  Bools are accepted (cast to 0/1) -- same as TB.
        if isinstance(scalar_value, str):
            raise ValueError(
                "could not convert "
                f"{type(scalar_value).__name__} to float: {scalar_value!r}")
        self.scalars.append((tag, scalar_value, global_step))

    def add_text(self, tag, text_string, global_step=None, **kw):
        self.add_text_calls += 1
        self.texts.append((tag, str(text_string), global_step))


# ---------------------------------------------------------------------------
# Test 1: diagnostics always contains a string-valued delta_mode.
# ---------------------------------------------------------------------------
def test_diagnostics_has_delta_mode():
    """`thin_surface_diagnostics()` returns a string `delta_mode` in both
    parameterizations.  This is the entry that triggered the CH8 crash."""
    print("\n--- Test 1: delta_mode is present and string-valued ---")
    for rel, expected_mode in [(False, "absolute"), (True, "relative")]:
        scene = _make_scene(n_points=4)
        args = _args(relative=rel, rho=0.5)
        scene.declare_optimizer(args, warmup=0, max_iterations=1000)
        scene.initialize_thin_surface(args, K=4)
        d = scene.thin_surface_diagnostics()
        assert d is not None
        assert "delta_mode" in d, f"missing delta_mode (rel={rel})"
        assert isinstance(d["delta_mode"], str), (
            f"delta_mode is str (got {type(d['delta_mode']).__name__})")
        assert d["delta_mode"] == expected_mode, (
            f"delta_mode={expected_mode} (got {d['delta_mode']!r})")
        # The remaining keys must be numeric (Python int/float, not str).
        non_numeric = [
            k for k, v in d.items()
            if k != "delta_mode" and not isinstance(v, (int, float))
        ]
        assert not non_numeric, (
            f"non-numeric non-mode entries: {non_numeric}")
        print(f"  [PASS] rel={rel}: delta_mode={d['delta_mode']!r}, "
              f"{len(d) - 1} numeric entries")


# ---------------------------------------------------------------------------
# Test 2: real SummaryWriter.add_scalar rejects a string (control).
# ---------------------------------------------------------------------------
def test_real_add_scalar_rejects_string():
    """Sanity: confirm the original CH8 failure mode is real and would
    be caught by the guard.  Uses a real tensorboard SummaryWriter
    writing to a tmp dir; just verifying the exception type."""
    print("\n--- Test 2: SummaryWriter.add_scalar rejects string ---")
    from torch.utils.tensorboard import SummaryWriter
    with tempfile.TemporaryDirectory() as td:
        w = SummaryWriter(td, purge_step=0)
        try:
            raised = False
            try:
                w.add_scalar("thin/delta_mode", "relative", 0)
            except ValueError as e:
                raised = True
                assert "could not convert" in str(e) or "float" in str(e), (
                    f"unexpected ValueError message: {e}")
            assert raised, "SummaryWriter.add_scalar must raise on string"
            print("  [PASS] add_scalar raises ValueError on string input")
        finally:
            w.close()


# ---------------------------------------------------------------------------
# Test 3: the fixed logging loop routes correctly.
# ---------------------------------------------------------------------------
def test_fixed_logging_loop_routes_correctly():
    """For both parameterizations the fixed loop must:
      - send every numeric entry to add_scalar
      - send delta_mode to add_text
      - never call add_scalar with a string
    """
    print("\n--- Test 3: fixed loop routes numeric -> add_scalar, "
          "string -> add_text ---")
    for rel in (False, True):
        scene = _make_scene(n_points=4)
        args = _args(relative=rel, rho=0.5)
        scene.declare_optimizer(args, warmup=0, max_iterations=1000)
        scene.initialize_thin_surface(args, K=4)
        d = scene.thin_surface_diagnostics()
        writer = _RecordingWriter()
        counts = _log_diag_dict(writer, d, step=42)
        # Exactly one text (delta_mode), rest scalars.
        assert counts["n_text"] == 1, (
            f"rel={rel}: exactly 1 text entry (got {counts['n_text']})")
        assert counts["n_text"] + counts["n_scalar"] == len(d), (
            f"rel={rel}: every diag key classified "
            f"(text+scale={counts['n_text']+counts['n_scalar']}, "
            f"diag has {len(d)})")
        assert writer.add_text_calls == 1, (
            f"rel={rel}: add_text called once (got {writer.add_text_calls})")
        assert writer.add_scalar_calls == len(d) - 1, (
            f"rel={rel}: add_scalar called {len(d)-1} times "
            f"(got {writer.add_scalar_calls})")
        # Every recorded add_scalar call has a numeric value.
        for tag, val, _ in writer.scalars:
            assert not isinstance(val, str), (
                f"string logged via add_scalar: {tag}={val!r}")
        # The text tag must be delta_mode and its value must be the mode.
        assert len(writer.texts) == 1, (
            f"rel={rel}: exactly one text record (got {len(writer.texts)})")
        tag, txt, step = writer.texts[0]
        assert tag == "thin/delta_mode", (
            f"rel={rel}: text tag is thin/delta_mode (got {tag!r})")
        assert txt == ("relative" if rel else "absolute"), (
            f"rel={rel}: text value matches mode (got {txt!r})")
        assert step == 42, f"step propagated (got {step})"
        print(f"  [PASS] rel={rel}: {writer.add_scalar_calls} scalars, "
              f"{writer.add_text_calls} text, no add_scalar on string")


# ---------------------------------------------------------------------------
# Test 4: real SummaryWriter accepts the routed dict end-to-end.
# ---------------------------------------------------------------------------
def test_real_summarywriter_accepts_full_dict():
    """End-to-end with a real SummaryWriter: the entire diagnostics dict
    must round-trip without raising.  Before the fix this crashed on the
    `delta_mode` key."""
    print("\n--- Test 4: real SummaryWriter end-to-end ---")
    from torch.utils.tensorboard import SummaryWriter
    for rel in (False, True):
        scene = _make_scene(n_points=4)
        args = _args(relative=rel, rho=0.5)
        scene.declare_optimizer(args, warmup=0, max_iterations=1000)
        scene.initialize_thin_surface(args, K=4)
        d = scene.thin_surface_diagnostics()
        with tempfile.TemporaryDirectory() as td:
            w = SummaryWriter(td, purge_step=0)
            try:
                _log_diag_dict(w, d, step=7)
                # No exception -- this is the regression assertion.
            finally:
                w.close()
        print(f"  [PASS] rel={rel}: real SummaryWriter accepted full dict")


# ---------------------------------------------------------------------------
# Test 5: a future diagnostic key with a non-numeric value is handled.
# ---------------------------------------------------------------------------
def test_fixed_loop_is_forward_compatible():
    """If the diagnostics dict ever gains a new non-numeric key (e.g. a
    future categorical), the loop must route it to add_text rather than
    raise.  Simulate by injecting a string key into a copy of the dict."""
    print("\n--- Test 5: forward-compat for future non-numeric keys ---")
    scene = _make_scene(n_points=4)
    args = _args(relative=True, rho=0.5)
    scene.declare_optimizer(args, warmup=0, max_iterations=1000)
    scene.initialize_thin_surface(args, K=4)
    d = dict(scene.thin_surface_diagnostics())
    d["future_categorical"] = "alpha"
    writer = _RecordingWriter()
    _log_diag_dict(writer, d, step=0)
    # Two text entries (delta_mode + future_categorical), rest scalars.
    assert writer.add_text_calls == 2, (
        f"two text entries (got {writer.add_text_calls})")
    assert writer.add_scalar_calls == len(d) - 2, (
        f"numeric entries all logged via add_scalar "
        f"(got {writer.add_scalar_calls})")
    tags = {tag for tag, _, _ in writer.texts}
    assert "thin/delta_mode" in tags
    assert "thin/future_categorical" in tags
    print("  [PASS] injected categorical routed to add_text, "
          "no exception raised")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
_any_failed = False


def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    global _any_failed
    if not cond:
        _any_failed = True


def _run_with_checks(fn, name):
    """Run `fn()`, intercepting AssertionErrors and converting to check().
    Lets us reuse the existing test bodies without rewriting them."""
    global _any_failed
    try:
        fn()
    except AssertionError as e:
        check(False, f"{name}: {e}")
    except Exception as e:
        check(False, f"{name}: unexpected {type(e).__name__}: {e}")


def main():
    print("=" * 60)
    print("Thin-Surface Diagnostics Logging Regression")
    print("=" * 60)

    _run_with_checks(test_diagnostics_has_delta_mode,
                     "diagnostics_has_delta_mode")
    _run_with_checks(test_real_add_scalar_rejects_string,
                     "real_add_scalar_rejects_string")
    _run_with_checks(test_fixed_logging_loop_routes_correctly,
                     "fixed_logging_loop_routes_correctly")
    _run_with_checks(test_real_summarywriter_accepts_full_dict,
                     "real_summarywriter_accepts_full_dict")
    _run_with_checks(test_fixed_loop_is_forward_compatible,
                     "fixed_loop_is_forward_compatible")

    print("\n" + "=" * 60)
    if _any_failed:
        print("SUMMARY: SOME TESTS FAILED.")
        sys.exit(1)
    print("SUMMARY: ALL DIAG-LOGGING TESTS PASSED.")


if __name__ == "__main__":
    main()