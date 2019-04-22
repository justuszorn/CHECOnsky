from setuptools import setup, find_packages

PACKAGENAME = "CHECOnsky"
DESCRIPTION = "Collection of resources for the on-sky campaigns of CHEC"
AUTHOR = "Jason J Watson"
AUTHOR_EMAIL = "jason.watson@physics.ox.ac.uk"
VERSION = "ASTRI1904"

setup(
    name=PACKAGENAME,
    packages=find_packages(),
    version=VERSION,
    description=DESCRIPTION,
    license='BSD3',
    install_requires=[
        'astropy',
        'scipy',
        'numpy',
        'matplotlib',
        'tqdm',
        'pandas>=0.21.0',
        'iminuit',
        'numba',
        'PyYAML',
        'seaborn',
        'CHECLabPy',
        'ctapipe',
    ],
    setup_requires=['pytest-runner', ],
    tests_require=['pytest', ],
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    package_data={
        '': ['calib/data/*'],
    },
    entry_points = {'console_scripts': [
        'extract_hillas = CHECOnskySB.scripts.extract_hillas:main',
        'merge_hillas = CHECOnskySB.scripts.merge_hillas:main',
    ]}
)
