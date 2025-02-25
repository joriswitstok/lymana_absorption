import subprocess
from setuptools import setup
from setuptools import find_packages

def readme():
    with open('README.md') as file:
        return(file.read())

setup(name='lymana_absorption',
      version='0.1',
      description='Spectral fitting of Lyman alpha absorption',
      long_description=readme(),
      classifiers=[
        'Development Status :: 0 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
      ],
      url='https://github.com/joriswitstok/lymana_absorption/',
      author='Joris Witstok',
      author_email='joris.witstok@nbi.ku.dk',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'matplotlib>=3.5.3',
        'numpy>=1.21.6',
        'scipy>=1.10.1',
        'astropy>=5.2',
        'spectres>=2.2.0',
        'corner',
        'seaborn',
        'pymultinest',
      ],
      python_requires='>=3.8',
     )
