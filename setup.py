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


core_requirements = [
    # version pinned in the consumer env (gymnasium==1.2.3); leave unpinned
    # so an already-installed gymnasium is accepted instead of force-reinstalled.
    "gymnasium",
    # pyquaternion doesn't support 2.x yet
    "numpy==1.26.*",
    "safetensors==0.3.3",
    # WARNING: recorded demos might break when updating Mujoco
    "mujoco",
    # needed for pyMJCF
    "dm_control",
    "imageio",
    "pyquaternion",
    "mujoco_utils",
    "wget",
    # mojo is installed from git by the consumer env; plain name here avoids a
    # forced reinstall of the already-present build.
    "mojo",
    "pyyaml",
    "dearpygui",
    "pyopenxr",
    # NOTE: mink (IK) is intentionally NOT a core dependency. mink requires
    # mujoco>=3.3.6, so installing bigym with deps would force-upgrade the
    # consumer env's pinned mujoco (recorded demos can break on a mujoco change,
    # see above). Install it separately instead, e.g.
    #   pip install --no-deps mink==1.1.0
    # It is exposed as the "ik" extra for convenience.
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
        # IK backend. Pulls mujoco>=3.3.6; install with --no-deps if you must
        # hold an older mujoco (see core_requirements note).
        "ik": ["mink"],
    },
)
