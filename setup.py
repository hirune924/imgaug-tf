from setuptools import setup, find_packages

setup(
    name='imgaugtf',
    version='0.1',
    packages=find_packages()
    package_dir={"": "src"},
    py_modules=["imgaugtf"],
)