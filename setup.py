# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt
======

Copyright © 2009-2010 CEA
Pierre Raybaut
Licensed under the terms of the CECILL License
(see guiqwt/__init__.py for details)
"""

# Building extensions:
# python setup.py build_ext -c mingw32 --inplace

from numpy.distutils.core import setup, Extension
import sys
import os
import os.path as osp
join = osp.join
from guidata.utils import get_subpackages, get_package_data

#TODO: copy qtdesigner plugins in Lib\site-packages\PyQt4\plugins\designer\python
#      note: this directory doesn't exist for a default PyQt4 install


LIBNAME = 'guiqwt'
import sys
from guiqwt import __version__ as version
# Remove module from list to allow building doc from build dir
del sys.modules['guiqwt']

DESCRIPTION = 'guiqwt is a set of tools for curve and image plotting (extension to PyQwt 5.2)'
LONG_DESCRIPTION = ''
KEYWORDS = ''
CLASSIFIERS = ['Development Status :: 5 - Production/Stable',
               'Topic :: Scientific/Engineering']

if os.name == 'nt':
    SCRIPTS = ['guiqwt-tests', 'guiqwt-tests.bat', 'sift', 'sift.bat']
else:
    SCRIPTS = ['guiqwt-tests', 'sift']
SCRIPTS = [join('scripts', fname) for fname in SCRIPTS]


try:
    import sphinx
except ImportError:
    sphinx = None
    
from numpy.distutils.command.build import build as dftbuild

class build(dftbuild):
    def has_doc(self):
        if sphinx is None:
            return False
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.isdir(os.path.join(setup_dir, 'doc'))
    sub_commands = dftbuild.sub_commands + [('build_doc', has_doc)]

cmdclass = {'build' : build}

if sphinx:
    from sphinx.setup_command import BuildDoc
    class build_doc(BuildDoc):
        def run(self):
            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version
            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))
            try:
                sphinx.setup_command.BuildDoc.run(self)
            except UnicodeDecodeError:
                print >>sys.stderr, "ERROR: unable to build documentation because Sphinx do not handle source path with non-ASCII characters. Please try to move the source package to another location (path with *only* ASCII characters)."            
            sys.path.pop(0)

    cmdclass['build_doc'] = build_doc


CFLAGS = ["-Wall"]
for arg, compile_arg in (("--sse2", "-msse2"),
                         ("--sse3", "-msse3"),):
    if arg in sys.argv:
        sys.argv.pop(sys.argv.index(arg))
        CFLAGS.insert(0, compile_arg)


setup(name=LIBNAME, version=version,
      download_url='http://%s.googlecode.com/files/%s-%s.zip' % (
                                                  LIBNAME, LIBNAME, version),
      description=DESCRIPTION, long_description=LONG_DESCRIPTION,
      packages=get_subpackages(LIBNAME),
      package_data={LIBNAME:
                    get_package_data(LIBNAME, ('.png', '.svg', '.mo', '.dcm',
                                               '.ui'))},
      requires=["PyQt4 (>4.3)", "NumPy", "guidata (>=1.3.0)"],
      scripts=SCRIPTS,
      ext_modules=[Extension(LIBNAME+'._ext', [join("src", 'histogram.f'),]),
                   Extension(LIBNAME+'._mandel', [join("src", 'mandel.f90')]),
                   Extension(LIBNAME+'._scaler', [join("src", "scaler.cpp"),
                                                  join("src", "pcolor.cpp")],
                             extra_compile_args=CFLAGS,
                             depends=[join("src", "traits.hpp"),
                                      join("src", "points.hpp"),
                                      join("src", "arrays.hpp"),
                                      join("src", "scaler.hpp"),
                                      join("src", "debug.hpp"),
                                      ],
                             ),
                   ],
      author = "Pierre Raybaut",
      author_email = 'pierre.raybaut@cea.fr',
      url = 'http://www.cea.fr',
      classifiers = CLASSIFIERS + [
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: OS Independent',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 2.6',
        ],
      cmdclass=cmdclass)
