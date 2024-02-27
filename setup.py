from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="irtorch",
    version="0.0.8",
    description="IRTorch: An item response theory package for python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    url="https://github.com/joakimwallmark/irtorch",
    author="Joakim Wallmark",
    author_email="wallmark.joakim@gmail.com",
    license="MIT",
    packages=find_packages(), #  finds all packages in the directory where the setup.py file is located
    package_data={
        "irtorch": [
            "datasets/national_mathematics/*.pt",
            "datasets/big_five/*.pt",
            # "datasets/swedish_sat/*.pt",
            # "datasets/swedish_sat/*.txt",
        ],
    },
    install_requires=[
        "torch",
        "factor_analyzer",
        "tensorboard",
        "numpy",
        "pandas",
        "plotly",
        "scikit-learn",
    ],
    zip_safe=False,
)
