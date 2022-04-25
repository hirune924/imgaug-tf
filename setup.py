from setuptools import setup, find_packages

setup(
    name='imgaugtf',
    version='0.1',
    packages=find_packages()
    url="https://github.com/hirune924/imgaug-tf/",
    package_dir={"": "src"},
    py_modules=["imgaugtf"]
)