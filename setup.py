from setuptools import setup, find_packages

setup(
    name='optw',
    version='1.0',
    entry_points={"console_scripts": ["optw=optw.__main__:main"]},
    packages=find_packages())