import numpy as np
from setuptools import setup
from setuptools import find_packages

VERSION = '1.1.1'

AUTHORS = 'Matthew Bourque, Lauren Chambers, Misty Cracraft, Mike Engesser, Mees Fix, Joe Filippazzo, Bryan Hilbert, '
AUTHORS += 'Graham Kanarek, Teagan King, Catherine Martlin, Shannon Osborne, Maria Pena-Guerrero, Johannes Sahlmann, '
AUTHORS += 'Ben Sunnquist, Brian York'

DESCRIPTION = 'The James Webb Space Telescope Quicklook Project'

REQUIRES = [
    'asdf',
    'astropy',
    'astroquery',
    'bandit',
    'bokeh<3',
    'codecov',
    'crds',
    'cryptography',
    'django<3.2',
    'flake8',
    'inflection',
    'ipython',
    'jinja2',
    'jsonschema',
    'jwst',
    'jwst_reffiles',
    'matplotlib',
    'nodejs',
    'numpy',
    'numpydoc',
    'pandas',
    'psycopg2',
    'pysiaf',
    'pytest',
    'pytest-cov',
    'pytest-mock',
    'pyvo',
    'scipy',
    'sphinx',
    'sphinx_rtd_theme',
    'sqlalchemy<2',
    'stdatamodels',
    'stsci_rtd_theme',
    'twine',
    'wtforms'
]

setup(
    name='jwql',
    version=VERSION,
    description=DESCRIPTION,
    url='https://github.com/spacetelescope/jwql.git',
    author=AUTHORS,
    author_email='jwql@stsci.edu',
    license='BSD',
    keywords=['astronomy', 'python'],
    classifiers=['Programming Language :: Python'],
    packages=find_packages(),
    install_requires=REQUIRES,
    include_package_data=True,
    include_dirs=[np.get_include()],
)
