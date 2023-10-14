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

import setuptools  # analysis:ignore
import numpy
import sys
import os
import os.path as osp
import subprocess
from numpy.distutils.core import setup, Extension


def get_package_data(name, extlist, exclude_dirs=[]):
    """
    Return data files for package *name* with extensions in *extlist*
    (search recursively in package directories)
    """
    assert isinstance(extlist, (list, tuple))
    flist = []
    # Workaround to replace os.path.relpath (not available until Python 2.6):
    offset = len(name) + len(os.pathsep)
    for dirpath, _dirnames, filenames in os.walk(name):
        if dirpath not in exclude_dirs:
            for fname in filenames:
                if osp.splitext(fname)[1].lower() in extlist:
                    flist.append(osp.join(dirpath, fname)[offset:])
    return flist


def get_subpackages(name):
    """Return subpackages of package *name*"""
    splist = []
    for dirpath, _dirnames, _filenames in os.walk(name):
        if osp.isfile(osp.join(dirpath, "__init__.py")):
            splist.append(".".join(dirpath.split(os.sep)))
    return splist


def cythonize_all(relpath):
    """Cythonize all Cython modules in relative path"""
    from Cython.Compiler import Main

    for fname in os.listdir(relpath):
        if osp.splitext(fname)[1] == ".pyx":
            Main.compile(osp.join(relpath, fname))


LIBNAME = "guiqwt"
__description__ = (
    "guiqwt is a set of tools for curve and image plotting (extension to PythonQwt)"
)

# Remove module from list to allow building doc from build dir
# del sys.modules["guiqwt"]

LONG_DESCRIPTION = """\
guiqwt: Python tools for curve and image plotting
=================================================

.. image:: https://raw.githubusercontent.com/PlotPyStack/guiqwt/master/doc/images/panorama.png

The guiqwt library is part of the `PlotPyStack`_ project, providing a set of
tools for creating GUIs for scientific/technical applications with Python,
Qt and SciPy/NumPy.

See `documentation`_ for more details on the library and `changelog`_ for
recent history of changes.

Copyright © 2009-2023 CEA, Pierre Raybaut, licensed under the terms of the
`CECILL License`_.

.. _documentation: https://guiqwt.readthedocs.io/en/latest/
.. _changelog: https://github.com/PlotPyStack/guiqwt/blob/master/CHANGELOG.md
.. _CECILL License: https://github.com/PlotPyStack/guiqwt/blob/master/Licence_CeCILL_V2-en.txt
.. _PlotPyStack: https://github.com/PlotPyStack


Overview
--------

Based on `PythonQwt`_ (a pure Python/PyQt reimplementation of the curve
plotting Qwt C++ library, included in guiqwt base source code) and on the
scientific modules NumPy and SciPy, ``guiqwt`` is a Python library providing
efficient 2D data-plotting features (curve/image visualization and related
tools) for interactive computing and signal/image processing application
development. It is based on Qt graphical user interfaces library, and
currently supports both ``PyQt5`` and ``PySide2``.

Extension to `PythonQwt`_:

* set of tools for curve and image plotting
* GUI-based application development helpers

.. _PythonQwt: https://pypi.python.org/pypi/PythonQwt


Building, installation, ...
---------------------------

The following packages are **required**: `PyQt5`_,
`PythonQwt`_, `guidata`_, `NumPy`_, `SciPy`_ and `Pillow`_.

.. _PyQt5: https://pypi.python.org/pypi/PyQt5
.. _PythonQwt: https://pypi.python.org/pypi/PythonQwt
.. _guidata: https://pypi.python.org/pypi/guidata
.. _NumPy: https://pypi.python.org/pypi/NumPy
.. _SciPy: https://pypi.python.org/pypi/SciPy
.. _Pillow: https://pypi.python.org/pypi/Pillow

See the `README`_ and `documentation`_ for more details.

.. _README: https://github.com/PlotPyStack/guiqwt/blob/master/README.md
"""

KEYWORDS = ""


def build_chm_doc(libname):
    """Return CHM documentation file (on Windows only), which is copied under
    {PythonInstallDir}\Doc, hence allowing Spyder to add an entry for opening
    package documentation in "Help" menu. This has no effect on a source
    distribution."""
    args = "".join(sys.argv)
    if "--no-doc" in sys.argv:
        sys.argv.remove("--no-doc")
        return
    if (
        os.name == "nt"
        and ("bdist" in args or "build" in args)
        and "--inplace" not in args
    ):
        try:
            import sphinx  # analysis:ignore
        except ImportError:
            print(
                "Warning: `sphinx` is required to build documentation", file=sys.stderr
            )
            return
        hhc_base = r"C:\Program Files%s\HTML Help Workshop\hhc.exe"
        for hhc_exe in (hhc_base % "", hhc_base % " (x86)"):
            if osp.isfile(hhc_exe):
                break
        else:
            print(
                "Warning: `HTML Help Workshop` is required to build CHM "
                "documentation file",
                file=sys.stderr,
            )
            return
        doctmp_dir = osp.join("build", "doctmp")
        if not osp.isdir(doctmp_dir):
            os.makedirs(doctmp_dir)
        fname = osp.abspath(osp.join(doctmp_dir, "%s.chm" % libname))
        if osp.isfile(fname):
            # doc has already been built
            return fname
        subprocess.call("sphinx-build -b htmlhelp doc %s" % doctmp_dir, shell=True)
        subprocess.call('"%s" %s' % (hhc_exe, fname), shell=True)
        if osp.isfile(fname):
            return fname
        else:
            print("Warning: CHM building process failed", file=sys.stderr)


CHM_DOC = build_chm_doc(LIBNAME)


def _create_script_list(basename):
    scripts = ["%s-py%d" % (basename, sys.version_info.major)]
    if os.name == "nt":
        scripts.append("%s.bat" % scripts[0])
    return [osp.join("scripts", name) for name in scripts]


SCRIPTS = _create_script_list("guiqwt-tests") + _create_script_list("sift")


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
    bld = dist.get_option_dict("build")
    if bld:
        comp = bld.get("compiler")
        if comp is not None and "mingw32" in comp:
            return False  # mingw is the compiler
    return os.name == "nt" and "mingw" not in "".join(sys.argv)


CFLAGS = ["-Wall"]
if is_msvc():
    CFLAGS.insert(0, "/EHsc")
for arg, compile_arg in (
    ("--sse2", "-msse2"),
    ("--sse3", "-msse3"),
):
    if arg in sys.argv:
        sys.argv.pop(sys.argv.index(arg))
        CFLAGS.insert(0, compile_arg)

# Compiling Cython modules to C source code: this is the only way I found to
# be able to build both Fortran and Cython extensions together
# (this could be changed now as there is no longer Fortran extensions here...)
cythonize_all("src")

setup(
    name=LIBNAME,
    version="4.4.3",  # Update here *AND* in __init__.py!
    # (Until setup.py has been fully retrofitted, this manual sync is mandatory)
    description=__description__,
    long_description=LONG_DESCRIPTION,
    packages=get_subpackages(LIBNAME),
    package_data={
        LIBNAME: get_package_data(LIBNAME, (".png", ".svg", ".mo", ".dcm", ".ui"))
    },
    data_files=[(r"Doc", [CHM_DOC])] if CHM_DOC else [],
    install_requires=[
        "NumPy>=1.3",
        "SciPy>=0.7",
        "guidata==3.1",
        "PythonQwt>=0.10",
        "Pillow",
        "QtPy>=1.3",
    ],
    extras_require={
        "Doc": ["Sphinx>=1.1"],
        "DICOM": ["pydicom>=0.9.3"],
    },
    entry_points={
        "gui_scripts": [
            "guiqwt-tests = guiqwt.tests:run",
            "sift = guiqwt.tests.sift:run",
        ]
    },
    ext_modules=[
        Extension(
            LIBNAME + ".histogram2d",
            [osp.join("src", "histogram2d.c")],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            LIBNAME + ".mandelbrot",
            [osp.join("src", "mandelbrot.c")],
            include_dirs=[numpy.get_include()],
        ),
        Extension(
            LIBNAME + "._scaler",
            [osp.join("src", "scaler.cpp"), osp.join("src", "pcolor.cpp")],
            extra_compile_args=CFLAGS,
            depends=[
                osp.join("src", "traits.hpp"),
                osp.join("src", "points.hpp"),
                osp.join("src", "arrays.hpp"),
                osp.join("src", "scaler.hpp"),
                osp.join("src", "debug.hpp"),
            ],
        ),
    ],
    author="Pierre Raybaut",
    author_email="pierre.raybaut@gmail.com",
    url="https://github.com/PlotPyStack/%s" % LIBNAME,
    license="CeCILL V2",
    classifiers=["Topic :: Scientific/Engineering"]
    + [
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: OS Independent",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
    ],
)
