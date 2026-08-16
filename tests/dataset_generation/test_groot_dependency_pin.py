from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GROOT_PY310_SONIC_REVISION = "ab88b50c718f6528e1df9dcbaf75865d1b604760"
GROOT_REQUIREMENT = (
    "gr00t @ git+https://github.com/NVIDIA/Isaac-GR00T.git@" f"{GROOT_PY310_SONIC_REVISION}"
)


def test_python_310_inference_extra_pins_sonic_capable_groot() -> None:
    pyproject = (REPO_ROOT / "gear_sonic" / "pyproject.toml").read_text(encoding="utf-8")
    installer = (REPO_ROOT / "install_scripts" / "install_inference.sh").read_text(encoding="utf-8")

    assert f'"{GROOT_REQUIREMENT}"' in pyproject
    assert 'git+https://github.com/NVIDIA/Isaac-GR00T.git"' not in pyproject
    assert "uv python install 3.10" in installer
    assert "uv python find --no-project 3.10" in installer
