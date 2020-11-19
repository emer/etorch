import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="etorch",
    version="1.1.0",
    author="emergent",
    author_email="oreilly@ucdavis.edu",
    description="eTorch integrates emergent GUI visualizations with PyTorch neural network framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/go-python/gopy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
