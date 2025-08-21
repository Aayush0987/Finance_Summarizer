from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> list[str]:
    """
    This function reads the requirements file and returns a list of packages.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
 

setup(
    name='Finance_Summarizer',
    version='0.0.1',
    author='Aayush Kumar',
    author_email='kumaraayush02@yahoo.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt'),
    description='A Python package for summarizing financial documents using LLMs.',
    long_description=open('README.md').read(),

)