from setuptools import setup, find_packages
import pathlib

# Read the requirements from requirements.txt
here = pathlib.Path(__file__).parent.resolve()
with open(here / 'requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    install_requires=requirements,  
)
