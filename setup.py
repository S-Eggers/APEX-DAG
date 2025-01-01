from setuptools import setup, find_packages

setup(
    name='ApexDAG',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'graphviz', # todo
    ],
    entry_points={
        'console_scripts': [
            # Define command-line scripts here
        ],
    },
)