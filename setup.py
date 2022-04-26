from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='imgaugtf',
    version='0.1',
    author="hirune924",
    description="tensorflow native image augmantation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hirune924/imgaug-tf/",
    packages=find_packages(),
    license="Apache License Version 2.0",
    install_requires=[
        "tensorflow >= 2.0",
        "tensorflow-addons >= 0.7.1",
        ],
    python_requires='>=3.4',
)