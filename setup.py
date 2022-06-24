from setuptools import setup
from pathlib import Path
from typing import Union

root_path = Path(__file__).resolve().parent

def get_requirements(file_path: Union[Path, str]):
    return [requirement.strip() for requirement in (root_path / file_path).open().readlines()]


setup(name='gym_mupen64plus',
      version='0.0.3',
      install_requires=get_requirements("requirements.txt"))
