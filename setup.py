from setuptools import setup, find_packages

setup(
	name="PyCrop",
	version='0.1.0dev',
	license="CC0 1.0 Universal",
	author="Benjamin A. Laken",
	author_email="benlaken@gmail.com",
	url="https://github.com/benlaken/PyCrop",
	packages=find_packages(exclude=['*test'])
	install_requires=['pandas','bokeh']
	)