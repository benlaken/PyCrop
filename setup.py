from setuptools import setup

setup(name="PyCrop",
      version='0.1.1dev',
      license="CC0 1.0 Universal",
      author="Benjamin A. Laken",
      author_email="benlaken@gmail.com",
      url="https://github.com/benlaken/PyCrop",
      packages=['PyCrop'],
      package_dir={'PyCrop': 'src/PyCrop'},
      package_data={'PyCrop': ['Data/*', 'examples/*']},
      install_requires=['pandas', 'bokeh'])
