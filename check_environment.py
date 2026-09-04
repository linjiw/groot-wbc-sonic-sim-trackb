#!/usr/bin/env python3
"""Pre-flight environment check for GR00T-WholeBodyControl.

Run this before training or deployment to verify all prerequisites are met.

Usage:
    python check_environment.py              # Check everything
    python check_environment.py --training   # Training checks only
    python check_environment.py --deploy     # C++ deploy checks only
    python check_environment.py --sim        # MuJoCo/Decoupled WBC sim checks only
"""

import importlib
from importlib.metadata import PackageNotFoundError, version as get_package_version
import os
import platform
import shutil
import subprocess
import sys


def check(name, passed, msg_pass="", msg_fail=""):
    symbol = "[+]" if passed else "[X]"
    detail = msg_pass if passed else msg_fail
    print(f"  {symbol} {name}: {detail}" if detail else f"  {symbol} {name}")
    return passed


def check_python(training=False):
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if training:
        ok = v.major == 3 and v.minor == 11
        return check(
            "Python version",
            ok,
            msg_pass=version_str,
            msg_fail=f"{version_str} (training requires 3.11.x — Isaac Lab requirement)",
        )
    else:
        ok = v.major == 3 and v.minor >= 10
        return check(
            "Python version",
            ok,
            msg_pass=version_str,
            msg_fail=f"{version_str} (need 3.10+)",
        )


def check_git_lfs():
    lfs_installed = shutil.which("git-lfs") is not None
    if not lfs_installed:
        return check("Git LFS", False, msg_fail="not installed (sudo apt install git-lfs)")

    # Check if LFS files are pulled (sample an actual LFS-tracked mesh file)
    mesh_path = "gear_sonic/data/assets/robot_description/urdf/g1/meshes"
    stl_files = []
    if os.path.isdir(mesh_path):
        stl_files = [os.path.join(mesh_path, f) for f in os.listdir(mesh_path) if f.endswith(".STL")]
    sample_file = stl_files[0] if stl_files else (
        "decoupled_wbc/sim2mujoco/resources/robots/g1/policy/"
        "GR00T-WholeBodyControl-Balance.onnx"
    )
    if os.path.exists(sample_file):
        size = os.path.getsize(sample_file)
        if size < 1000:
            return check(
                "Git LFS",
                False,
                msg_fail=f"{sample_file} is {size} bytes (LFS pointer — run 'git lfs pull')",
            )
        return check("Git LFS", True, msg_pass="installed, files pulled")
    return check("Git LFS", True, msg_pass="installed")


def check_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return check("CUDA", True, msg_pass=f"{device_name} (CUDA {cuda_version})")
        else:
            return check("CUDA", False, msg_fail="torch.cuda.is_available() = False")
    except ImportError:
        return check("CUDA", False, msg_fail="PyTorch not installed")


def check_torch():
    try:
        import torch

        return check("PyTorch", True, msg_pass=torch.__version__)
    except ImportError:
        return check(
            "PyTorch",
            False,
            msg_fail="not installed (pip install torch)",
        )


def check_isaaclab():
    try:
        import isaaclab

        version = getattr(isaaclab, "__version__", "unknown")
        return check("Isaac Lab", True, msg_pass=version)
    except ImportError:
        return check(
            "Isaac Lab",
            False,
            msg_fail="not installed — see https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html",
        )


def check_gear_sonic():
    try:
        from importlib.metadata import version as get_version

        ver = get_version("gear_sonic")
        return check("gear_sonic", True, msg_pass=f"installed ({ver})")
    except (ImportError, PackageNotFoundError):
        return check(
            "gear_sonic",
            False,
            msg_fail="not installed (pip install -e 'gear_sonic[training]')",
        )


def check_training_deps():
    results = []
    for pkg, pip_name in [
        ("hydra", "hydra-core"),
        ("tensordict", "tensordict"),
        ("trl", "trl"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("wandb", "wandb"),
        ("tensorboard", "tensorboard"),
        ("open3d", "open3d"),
        ("vector_quantize_pytorch", "vector-quantize-pytorch"),
        ("smpl_sim", "smpl_sim"),
    ]:
        try:
            mod = importlib.import_module(pkg)
            package_version = getattr(mod, "__version__", None)
            if package_version is None:
                try:
                    package_version = get_package_version(pip_name)
                except PackageNotFoundError:
                    package_version = "ok"
            results.append(check(pip_name, True, msg_pass=package_version))
        except ImportError:
            results.append(
                check(pip_name, False, msg_fail=f"not installed (pip install {pip_name})")
            )
    return all(results)


def check_tensorrt():
    trt_root = os.environ.get("TensorRT_ROOT", "")
    if not trt_root:
        return check(
            "TensorRT",
            False,
            msg_fail="TensorRT_ROOT not set (export TensorRT_ROOT=$HOME/TensorRT)",
        )
    if not os.path.isdir(trt_root):
        return check("TensorRT", False, msg_fail=f"TensorRT_ROOT={trt_root} does not exist")

    # Check for the library
    lib_dir = os.path.join(trt_root, "lib")
    if os.path.isdir(lib_dir):
        libs = [f for f in os.listdir(lib_dir) if "nvinfer" in f and ".so" in f]
        if libs:
            # Try to extract version from filename
            for lib in libs:
                if "nvinfer.so." in lib:
                    version = lib.split("nvinfer.so.")[-1]
                    return check("TensorRT", True, msg_pass=f"{version} at {trt_root}")
            return check("TensorRT", True, msg_pass=f"found at {trt_root}")

    return check("TensorRT", False, msg_fail=f"libnvinfer not found in {lib_dir}")


def check_command_available(command, install_hint):
    path = shutil.which(command)
    if path:
        return check(command, True, msg_pass=path)
    return check(command, False, msg_fail=f"not found ({install_hint})")


def check_cuda_toolkit():
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            output = subprocess.check_output([nvcc, "--version"], text=True, stderr=subprocess.STDOUT)
            version_line = output.strip().splitlines()[-1]
        except Exception:
            version_line = nvcc
        return check("CUDA toolkit (nvcc)", True, msg_pass=version_line)

    for root in [
        os.environ.get("CUDAToolkit_ROOT"),
        os.environ.get("CUDA_HOME"),
        "/usr/local/cuda",
        "/usr/local/cuda-13.0",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.4",
        "/usr/local/cuda-12",
    ]:
        if not root:
            continue
        candidate = os.path.join(root, "bin", "nvcc")
        if os.path.exists(candidate):
            return check("CUDA toolkit (nvcc)", True, msg_pass=candidate)

    return check(
        "CUDA toolkit (nvcc)",
        False,
        msg_fail=(
            "not found (install cuda-toolkit matching TensorRT; "
            "PyTorch CUDA runtime alone is not enough for C++ builds)"
        ),
    )


def check_onnxruntime_cmake():
    candidates = []
    root_dir = os.environ.get("onnxruntime_ROOT")
    if root_dir:
        candidates.append(root_dir)
    env_dir = os.environ.get("onnxruntime_DIR")
    if env_dir:
        candidates.append(env_dir)
        # Existing deploy scripts sometimes set onnxruntime_DIR to
        # <root>/lib/cmake/onnxruntime, while this repo's custom CMake finder
        # searches roots with include/ and lib/ suffixes.
        cmake_suffix = os.path.join("lib", "cmake", "onnxruntime")
        if env_dir.endswith(cmake_suffix):
            candidates.append(os.path.dirname(os.path.dirname(os.path.dirname(env_dir))))
    candidates.extend(
        [
            "/opt/onnxruntime",
            "/usr/local/onnxruntime",
            os.path.expanduser("~/.local/onnxruntime"),
        ]
    )
    for root in candidates:
        if not root or not os.path.isdir(root):
            continue
        include = os.path.join(root, "include", "onnxruntime_cxx_api.h")
        lib_dir = os.path.join(root, "lib")
        if os.path.exists(include) and os.path.isdir(lib_dir):
            libs = [name for name in os.listdir(lib_dir) if name.startswith("libonnxruntime.so")]
            if libs:
                return check("ONNX Runtime C/C++", True, msg_pass=root)
    return check(
        "ONNX Runtime C/C++",
        False,
        msg_fail="not found (install ONNX Runtime C/C++ package or set onnxruntime_ROOT)",
    )


def check_deploy_assets():
    required = [
        "gear_sonic_deploy/policy/release/model_encoder.onnx",
        "gear_sonic_deploy/policy/release/model_decoder.onnx",
        "gear_sonic_deploy/policy/release/observation_config.yaml",
        "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
        "gear_sonic_deploy/reference/example",
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        return check("Deploy assets", False, msg_fail="missing: " + ", ".join(missing))
    pointer_like = [path for path in required if path.endswith(".onnx") and os.path.getsize(path) < 1000]
    if pointer_like:
        return check(
            "Deploy assets",
            False,
            msg_fail="LFS pointer or truncated files: " + ", ".join(pointer_like),
        )
    return check(
        "Deploy assets",
        True,
        msg_pass="release ONNXs, observation config, planner, and reference data present",
    )


def check_deploy_toolchain():
    results = [
        check_command_available("cmake", "sudo apt install cmake"),
        check_command_available("ninja", "sudo apt install ninja-build"),
        check_command_available("just", "sudo apt install just OR cargo install just"),
        check_command_available("clang", "sudo apt install clang"),
        check_cuda_toolkit(),
        check_onnxruntime_cmake(),
        check_deploy_assets(),
    ]
    return all(results)


def check_sim_deps():
    results = []
    for module, install_hint in [
        ("onnxruntime", "bash install_scripts/install_mujoco_sim.sh"),
        ("mujoco", "bash install_scripts/install_mujoco_sim.sh"),
        ("decoupled_wbc", "bash install_scripts/install_mujoco_sim.sh"),
    ]:
        try:
            importlib.import_module(module)
            results.append(check(module, True, msg_pass="installed"))
        except ImportError:
            results.append(check(module, False, msg_fail=f"not installed ({install_hint})"))
    return all(results)


def check_disk_space():
    stat = os.statvfs(".")
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    ok = free_gb > 10
    return check(
        "Disk space",
        ok,
        msg_pass=f"{free_gb:.0f} GB free",
        msg_fail=f"{free_gb:.1f} GB free (recommend 10+ GB)",
    )


def main():
    mode = "all"
    if "--training" in sys.argv:
        mode = "training"
    elif "--deploy" in sys.argv:
        mode = "deploy"
    elif "--sim" in sys.argv or "--mujoco" in sys.argv:
        mode = "sim"

    print("GR00T-WholeBodyControl Environment Check")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python:   {sys.executable}")
    print()

    all_pass = True

    # Basic checks (always run)
    print("Basic:")
    all_pass &= check_python(training=(mode in ("all", "training")))
    all_pass &= check_git_lfs()
    all_pass &= check_cuda()
    all_pass &= check_torch()
    all_pass &= check_disk_space()
    print()

    if mode == "sim":
        print("Simulation:")
        all_pass &= check_sim_deps()
        print()

    if mode in ("all", "training"):
        print("Training:")
        all_pass &= check_isaaclab()
        all_pass &= check_gear_sonic()
        all_pass &= check_training_deps()
        print()

    if mode in ("all", "deploy"):
        print("Deployment:")
        all_pass &= check_tensorrt()
        all_pass &= check_deploy_toolchain()
        print()

    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks failed. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
