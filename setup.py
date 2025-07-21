# setup.py
from setuptools import setup, find_packages

setup(
    name='dix_inversion',
    version='0.1',
    description='Rayleigh wave inversion using Dix method',
    author='Prahlada Mittal',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
)