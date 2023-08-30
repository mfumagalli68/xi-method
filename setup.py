import distutils.text_file
from pathlib import Path

from setuptools import setup, find_packages


def parse_requirements(filename):

    file = str(Path(__file__).with_name(filename))
    return distutils.text_file.TextFile(filename=file).readlines()


setup(name='xi_method-method',
      version='0.1.0',
      python_requires=">=3.8",
      description='Xi - Post hoc explanations',
      url='https://github.com/mfumagalli68/xi/tree/main',
      author='Marco fumagalli',
      author_email='m.fumagalli68@gmail.com',
      packages=find_packages(),
      license='MIT',
      install_requires=
      parse_requirements('requirements.txt'),
      zip_safe=False,
      include_package_data=True)

