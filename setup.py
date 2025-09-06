#!/usr/bin/env python3
"""
vLLM Cluster Setup Script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def parse_requirements(filename):
    """Parse requirements from file, handling -r includes"""
    requirements = []
    with open(filename, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.startswith("-r"):
                    # Handle -r requirements-core.txt
                    included_file = line.split()[1]
                    requirements.extend(parse_requirements(included_file))
                else:
                    requirements.append(line)
    return requirements

requirements = parse_requirements("requirements-core.txt")

setup(
    name="vllm-cluster",
    version="2.0.0",
    author="vLLM Cluster Team",
    author_email="team@vllm-cluster.org",
    description="Production-ready distributed vLLM inference cluster",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/vllm-cluster",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": parse_requirements("requirements-dev.txt"),
        "evaluation": parse_requirements("requirements-eval.txt"),
        "all": parse_requirements("requirements-dev.txt") + parse_requirements("requirements-eval.txt"),
    },
    entry_points={
        "console_scripts": [
            "vllm-cluster=vllm_cluster.cli.main:main",
            "vllm-single=vllm_standalone.cli.main:main",
        ],
    },
    scripts=[
        "vllm-cluster",
        "vllm-single"
    ],
    include_package_data=True,
    package_data={
        "vllm_cluster": [
            "configs/templates/*.yaml",
            "configs/examples/*.yaml",
        ],
    },
)