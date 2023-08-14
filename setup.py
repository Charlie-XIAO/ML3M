import os

from setuptools import find_packages, setup

import ml3m

VERSION = ml3m.__version__
DESCRIPTION = "Evaluting your LLM performance"

dirname = os.path.dirname(__file__)
readme_path = os.path.join(dirname, "README.md")
with open(readme_path, "r", encoding="utf-8") as f:
    README = f.read()

setup(
    name="ml3m",
    version=VERSION,
    author="Charlie-XIAO (Yao Xiao)",
    author_email="yx2436@nyu.edu",
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["pandas", "numpy", "openai", "tqdm"],
    keywords=["python", "LLM", "evaluation"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
    ],
)
