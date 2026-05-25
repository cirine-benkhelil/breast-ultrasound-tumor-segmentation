from setuptools import setup, find_packages

setup(
    name="bus-ai",
    version="1.0.0",
    description="Deep learning pipeline for breast tumor segmentation and classification on ultrasound images",
    author="[Your Name]",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "matplotlib>=3.6.0",
        "opencv-python>=4.7.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "segmentation-models-pytorch==0.3.3",
        "albumentations>=1.3.0",
        "scikit-learn>=1.2.0",
        "timm>=0.9.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
