from setuptools import setup, find_packages

setup(
    name="gce",
    version="1.0.0",
    description="Generative Causal Explainers (GCE) for Parkinson's disease neural data",
    url="https://github.com/tanmoysil/GCE_release",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "statsmodels>=0.14",
        "torch>=2.0",
        "captum>=0.7",
        "dcor>=0.6",
        "h5py>=3.8",
        "openpyxl>=3.1",
        "tensorboard>=2.13",
        "PyYAML>=6.0",
    ],
)
