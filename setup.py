import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="st_toolbox-sekro", # Replace with your own username
    version="2021.0.1",
    author="Sebastian Krossa",
    author_email="sebastian.krossa@ntnu.no",
    description="A toolbox for working with spatial transcriptomics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sekro/spatial_transcriptomics_toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)