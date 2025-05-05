from setuptools import setup, find_packages

__version__ = "0.1.0"

setup(
    name = "icon",
    version=__version__,
    description="Intra Contrast (ICon).",
    author="Junlin Wang",
    author_email="12112921@mail.sustech.edu.cn",
    url="https://github.com/HenryWJL/icon",
    packages=[
        package for package in find_packages() 
        if package.startswith("icon")
    ],
    python_requires=">=3.9",
    setup_requires=["setuptools>=62.3.0"],
    include_package_data=True,
    install_requires=[
        "click>=8.1.7",
        "imageio>=2.34.0",
        "imageio-ffmpeg>=0.4.9",
        "zarr>=2.17.0",
        "numpy>=1.24.0,<1.27.0",
        "scipy>=1.13.1",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "timm>=0.9.7",
        "diffusers>=0.27.2",
        "tqdm>=4.67.1",
        "hydra-core>=1.2.0",
        "omegaconf>=2.3.0",
        "einops>=0.8.0",
        "wandb>=0.16.3",
        "numba>=0.60.0",
        "gymnasium>=1.0.0",
        "mani-skill>=0.3.0",
    ]
)
