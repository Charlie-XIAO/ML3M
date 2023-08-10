import os

from setuptools import find_packages, setup

VERSION = "0.0.3.dev"
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
    install_requires=["pandas", "numpy", "openai"],
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
