# -*- coding: utf-8 -*-
#
# Copyright © 2009-2015 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt
======

Copyright © 2009-2015 CEA
Pierre Raybaut
Licensed under the terms of the CECILL License
(see guiqwt/__init__.py for details)
"""

# Building extensions:
# python setup.py build_ext -c mingw32 --inplace

from __future__ import print_function

import setuptools  # analysis:ignore
import numpy
import sys
import os
import os.path as osp
from numpy.distutils.core import setup, Extension
from guidata.utils import get_subpackages, get_package_data, cythonize_all

#TODO: copy qtdesigner plugins in Lib\site-packages\PyQt4\plugins\designer\python
#      note: this directory doesn't exist for a default PyQt4 install


LIBNAME = 'guiqwt'
from guiqwt import __version__ as version
# Remove module from list to allow building doc from build dir
del sys.modules['guiqwt']

DESCRIPTION = 'guiqwt is a set of tools for curve and image plotting '\
              '(extension to `PythonQwt`)'
LONG_DESCRIPTION = ''
KEYWORDS = ''
CLASSIFIERS = ['Topic :: Scientific/Engineering']
if 'beta' in version or 'b' in version:
    CLASSIFIERS += ['Development Status :: 4 - Beta']
elif 'alpha' in version or 'a' in version:
    CLASSIFIERS += ['Development Status :: 3 - Alpha']
else:
    CLASSIFIERS += ['Development Status :: 5 - Production/Stable']


def _create_script_list(basename):
    scripts = ['%s-py%d' % (basename, sys.version_info.major)]
    if os.name == 'nt':
        scripts.append('%s.bat' % scripts[0])
    return [osp.join('scripts', name) for name in scripts]

SCRIPTS = _create_script_list('guiqwt-tests') + _create_script_list('sift')


try:
    import sphinx
except ImportError:
    sphinx = None  # analysis:ignore
    

def is_msvc():
    """Detect if Microsoft Visual C++ compiler was chosen to build package"""
    # checking if mingw is the compiler
    # mingw32 compiler configured in %USERPROFILE%\pydistutils.cfg 
    # or distutils\distutils.cfg
    from distutils.dist import Distribution
    dist = Distribution()
    dist.parse_config_files()
    bld = dist.get_option_dict('build')
    if bld:
        comp = bld.get('compiler')
        if comp is not None and 'mingw32' in comp:
            return False  # mingw is the compiler
    return os.name == 'nt' and 'mingw' not in ''.join(sys.argv)

CFLAGS = ["-Wall"]
if is_msvc():
    CFLAGS.insert(0, "/EHsc")
for arg, compile_arg in (("--sse2", "-msse2"),
                         ("--sse3", "-msse3"),):
    if arg in sys.argv:
        sys.argv.pop(sys.argv.index(arg))
        CFLAGS.insert(0, compile_arg)

# Compiling Cython modules to C source code: this is the only way I found to 
# be able to build both Fortran and Cython extensions together
# (this could be changed now as there is no longer Fortran extensions here...)
cythonize_all('src')

setup(name=LIBNAME, version=version,
      description=DESCRIPTION, long_description=LONG_DESCRIPTION,
      packages=get_subpackages(LIBNAME),
      package_data={LIBNAME:
                    get_package_data(LIBNAME, ('.png', '.svg', '.mo', '.dcm',
                                               '.ui'))},
      install_requires=["NumPy>=1.3", "SciPy>=0.7", "guidata>=1.7.0",
                        "PythonQwt>=0.5.0", "Pillow"],
      extras_require = {
                        'Doc':  ["Sphinx>=1.1"],
                        },
      entry_points={'gui_scripts':
                    ['guiqwt-tests-py%d = guiqwt.tests:run'\
                     % sys.version_info.major,
                     'sift-py%d = guiqwt.tests.sift:run'\
                     % sys.version_info.major,]},
      ext_modules=[Extension(LIBNAME+'.histogram2d',
                             [osp.join('src', 'histogram2d.c')],
                             include_dirs=[numpy.get_include()]),
                   Extension(LIBNAME+'.mandelbrot',
                             [osp.join('src', 'mandelbrot.c')],
                             include_dirs=[numpy.get_include()]),
                   Extension(LIBNAME+'._scaler',
                             [osp.join("src", "scaler.cpp"),
                              osp.join("src", "pcolor.cpp")],
                             extra_compile_args=CFLAGS,
                             depends=[osp.join("src", "traits.hpp"),
                                      osp.join("src", "points.hpp"),
                                      osp.join("src", "arrays.hpp"),
                                      osp.join("src", "scaler.hpp"),
                                      osp.join("src", "debug.hpp"),
                                      ],
                             ),
                   ],
      author = "Pierre Raybaut",
      author_email = 'pierre.raybaut@gmail.com',
      url = 'https://github.com/PierreRaybaut/%s' % LIBNAME,
      license = 'CeCILL V2',
      classifiers = CLASSIFIERS + [
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        ],
      )
