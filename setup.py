"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the version
with open('version.txt', 'r') as version_file:
    current_version = version_file.read().rstrip()

# Packages used
documentation_packages = [
    "sphinx==1.6b3",
    "sphinxcontrib-napoleon==0.6.1",
    "sphinxcontrib-programoutput==0.10",
    "sphinxcontrib-websupport==1.0.1"
]
regular_packages = [
    'numpy',
    'scipy',
    'setuptools'
]
testing_packages = ["coverage==4.4.1"]

setup(
    name='cheshire',
    version=current_version,
    description='Solve the 2D Schrodinger equation',
    long_description=long_description,
    # url='https://github.com/pypa/sampleproject', 
    author='James McElveen',
    author_email='jmcelve2@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Physicists',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='quantum numpy scipy',
    packages=find_packages(), 
    install_requires=[regular_packages +
                      documentation_packages +
                      testing_packages],
    extras_require={
        'doc': documentation_packages,
        'all': regular_packages + documentation_packages + testing_packages,
    },
    zip_safe=False,
    test_suite="tests"
    )
