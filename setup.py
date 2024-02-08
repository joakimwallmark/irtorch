from setuptools import setup, find_packages

setup(
    name="irtorch",
    version="0.0.3",
    description="A python package for item response theory.",
    long_description="README.md",
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    url="",
    author="Joakim Wallmark",
    author_email="wallmark.joakim@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "factor_analyzer",
        "tensorboard",
        "tqdm",
        "feather-format",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
    ],
    zip_safe=False,
)
