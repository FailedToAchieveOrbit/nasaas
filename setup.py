"""
Setup configuration for NASaaS package
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nasaas",
    version="0.1.0",
    author="FailedToAchieveOrbit",
    author_email="sawadall@asu.edu",
    description="Neural Architecture Search as a Service - Autonomous neural network design from natural language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FailedToAchieveOrbit/nasaas",
    project_urls={
        "Bug Reports": "https://github.com/FailedToAchieveOrbit/nasaas/issues",
        "Source": "https://github.com/FailedToAchieveOrbit/nasaas",
        "Documentation": "https://nasaas.readthedocs.io",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipywidgets>=8.1.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.13.0",
            "plotly>=5.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nasaas=nasaas.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="neural-architecture-search, nas, machine-learning, ai, deep-learning, pytorch, automation",
)