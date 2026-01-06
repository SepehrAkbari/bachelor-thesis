from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="bachelor_thesis",
    packages=find_packages(),
    python_requires='>=3.7',
    ext_modules=cythonize("src/include/wrapped.pyx"),
)