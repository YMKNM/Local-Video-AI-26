"""
Video AI Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="video-ai",
    version="0.1.0",
    author="Video AI Project",
    description="Local AI video generation system for AMD GPUs on Windows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-ai",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'video_ai': [
            'configs/*.yaml',
        ],
    },
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'isort>=5.12.0',
            'mypy>=1.0.0',
        ],
        'api': [
            'fastapi>=0.100.0',
            'uvicorn>=0.23.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'video-ai=generate:main',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="video generation, ai, amd, directml, onnx, diffusion",
)
