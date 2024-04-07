from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='modaverse',
      version='0.0.1',
      author='Xinyu Wang',
      author_email='xinyu.wang02@adelaide.edu.au',
      packages=find_packages(),
      install_requires=required)
