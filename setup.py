from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(name='optimal_control',
      version='0.1',
      description='A implementation of a regression based algorithm for optimal control problems.',
      url='https://bitbucket.org/hagerpa/optimal-control/',
      author='Paul Hager',
      author_email='hagerpa@gmail.com',
      license='MIT',
      packages=['optimal_control'],
      ext_modules=cythonize([
          "optimal_control/basis/memory_efficient_product.pyx",
      ]),
      include_dirs=[numpy.get_include()],
      install_requires=['Cython', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas'],
      zip_safe=False)
