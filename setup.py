from setuptools import find_packages, setup

setup(
    name='src',
    python_requires='~=3.12',
    packages=find_packages(),
    version='0.1.0',
    description='Gravitational-wave noise modelling with autoencoders',
    author='Ronaldas Macas',
    license='MIT',
)
