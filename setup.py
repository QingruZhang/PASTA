from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

install_requires = [
    "transformers>=4.31.1"
]

setup(
    name="pastalib",
    version="0.1.3",
    author="Qingru Zhang, Chandan Singh, Lucas Liu, Xiaodong Liu, Bin Yu, Jianfeng Gao, Tuo Zhao",
    author_email="qingru.zhang@gatech.edu",
    description="PyTorch Implementation for PASTA, A Post-hoc Attention Steering Approach that enables users to emphasize specific contexts for LLMs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QingruZhang/PASTA",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.7.0',
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)