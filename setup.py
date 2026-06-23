import codecs
import os
from pathlib import Path

import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


# Pins are exact and self-contained so `pip install bigym` works standalone,
# without a pre-built consumer env. The versions match the MoF research env
# (https://github.com/pointW/equidiff) so the two installs never conflict.
core_requirements = [
    # bigym runs against stock gymnasium 1.2.x (no stepjam fork needed).
    "gymnasium==1.2.3",
    # pyquaternion doesn't support numpy 2.x yet.
    "numpy==1.26.*",
    # demo (de)serialization. 0.7.x reads the recorded 0.3.x safetensors fine.
    "safetensors==0.7.0",
    # WARNING: recorded demos might replay differently if Mujoco changes, so the
    # version is held. mink (IK, "ik" extra) wants mujoco>=3.3.6 and so is kept
    # out of core to avoid force-upgrading this pin.
    "mujoco==3.3.5",
    # needed for pyMJCF
    "dm_control==1.0.31",
    "imageio==2.22.0",
    "pyquaternion==0.9.9",
    "mujoco_utils==0.0.6",
    "wget==3.2",
    # PyPI "mojo" is a different package (Modular's language); the env's bigym
    # needs stepjam's mojo, which is only on git.
    "mojo @ git+https://github.com/stepjam/mojo.git@0.1.1",
    "pyyaml==6.0.3",
]

setuptools.setup(
    version=get_version("bigym/__init__.py"),
    name="bigym",
    author="Nikita Cherniadev",
    author_email="nikita.chernyadev@gmail.com",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=core_requirements,
    package_data={
        "": [str(p.resolve()) for p in Path("bigym/envs/xmls").glob("**/*")]
        + [str(p.resolve()) for p in Path("bigym/envs/presets").glob("**/*.yaml")]
        + [str(p.resolve()) for p in Path("vr/viewer/xmls").glob("**/*")]
    },
    extras_require={
        "dev": ["pre-commit", "pytest"],
        "examples": [
            "moviepy",
            "pygame",
            "opencv-python",
            "matplotlib",
        ],
        # IK backend. Pulls mujoco>=3.3.6, which would force-upgrade the core
        # mujoco==3.3.5 pin; install with --no-deps if you must hold mujoco.
        "ik": ["mink==1.1.0"],
        # Desktop GUI tools (demo player/recorder under tools/).
        "tools": ["dearpygui==2.3.1"],
        # VR teleoperation viewer (vr/).
        "vr": ["pyopenxr==1.1.5301"],
    },
)
