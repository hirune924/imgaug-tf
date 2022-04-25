from setuptools import setup, find_packages

setup(
    name='imgaugtf',
    version='0.1',
    #url="https://github.com/hirune924/imgaug-tf/",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=["imgaugtf"]
)