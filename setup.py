from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="irtorch",
    version="0.4.4",
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
            "datasets/big_five/*.txt",
            "datasets/swedish_sat/*.pt",
            "datasets/swedish_sat/*.txt",
        ],
    },
    install_requires=[
        "torch>=2.1.1,<4.0.0",
        "numpy>=1.24.1,<3.0.0",
        "pandas>=2.2.0,<4.0.0",
        "plotly>=5.19.0,<7.0.0",
        "factor_analyzer>=0.4.1",
    ],
    zip_safe=False,
)
