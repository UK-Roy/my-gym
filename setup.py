import os

from setuptools import find_packages, setup

with open(os.path.join("mygym", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="mygym",
    description="Set of robotic environments based on PyBullet physics engine and gymnasium.",
    author="Utsha Kumar Roy",
    author_email="utsharoy99@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UK-Roy/my_gym",
    packages=find_packages(),
    include_package_data=True,
    package_data={"mygym": ["version.txt"]},
    version=__version__,
    install_requires=["gymnasium>=0.26", "pybullet", "numpy", "scipy"],
    extras_require={
        "develop": ["pytest-cov", "black", "isort", "pytype", "sphinx", "sphinx-rtd-theme"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
