import pathlib
import re

import setuptools


HERE = pathlib.Path(__file__).resolve().parent

name = "sigformer"

with open(HERE / name / "__init__.py") as f:
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if match:
        version = match.group(1)
    else:
        RuntimeError("Cannot find version string")

python_requires = "~=3.7"
install_requires = ["jax", "equinox", "signax"]


setuptools.setup(
    name=name,
    version=version,
    install_requires=install_requires,
    packages=setuptools.find_packages(),
)
