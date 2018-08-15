# setup.py
from setuptools import setup, find_packages

setup(name='sentiment',
  version='0.1',
  packages=find_packages(),
  description='example to run keras on gcloud ml-engine',
  author='Uyen Hoang',
  author_email='uyen@example.com',
  license='MIT',
  install_requires=[
      'keras==2.1.5',
      'h5py',
      'numpy'
  ],
  zip_safe=False)

