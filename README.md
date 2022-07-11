# guiqwt: Python tools for curve and image plotting

[![license](https://img.shields.io/pypi/l/guiqwt.svg)](./LICENSE)
[![pypi version](https://img.shields.io/pypi/v/guiqwt.svg)](https://pypi.org/project/guiqwt/)
[![PyPI status](https://img.shields.io/pypi/status/guiqwt.svg)](https://github.com/PierreRaybaut/guiqwt)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/guiqwt.svg)](https://pypi.python.org/pypi/guiqwt/)
[![download count](https://img.shields.io/conda/dn/conda-forge/guiqwt.svg)](https://www.anaconda.com/download/)

<img src="https://raw.githubusercontent.com/PierreRaybaut/guiqwt/master/doc/images/panorama.png">

See [documentation](https://guiqwt.readthedocs.io/en/latest/) for more details on
the library and [changelog](CHANGELOG.md) for recent history of changes.

Copyright © 2009-2015 CEA, Pierre Raybaut, licensed under the terms of the
[CECILL License](Licence_CeCILL_V2-en.txt).

## Overview

Based on ``PythonQwt`` (a pure Python/PyQt reimplementation of the curve
plotting Qwt C++ library, included in guiqwt base source code) and on the
scientific modules NumPy and SciPy, ``guiqwt`` is a Python library providing
efficient 2D data-plotting features (curve/image visualization and related
tools) for interactive computing and signal/image processing application
development. It is based on Qt graphical user interfaces library, and
currently supports both ``PyQt5`` and ``PySide2``.

Extension to ``PythonQwt``:

* set of tools for curve and image plotting
* GUI-based application development helpers

## Dependencies

### Requirements

* Python 3.7+
* [PyQt5](https://pypi.python.org/pypi/PyQt5)
* [QtPy](https://pypi.org/project/QtPy/)
* [PythonQwt](https://pypi.python.org/pypi/PythonQwt) (pure Python reimplementation of Qwt6 C++ library)
* [guidata](https://pypi.python.org/pypi/guidata)
* [NumPy](https://pypi.python.org/pypi/NumPy)
* [SciPy](https://pypi.python.org/pypi/SciPy)
* [Pillow](https://pypi.python.org/pypi/Pillow)

### Optional modules

* [pydicom](https://pypi.python.org/pypi/pydicom) for DICOM I/O support

## Building/Installation

### All platforms

The setup.py script supports the following extra options for
optimizing the image scaler engine with SSE2/SSE3 processors:
``--sse2`` or ``--sse3``.

### On GNU/Linux and MacOS platforms

```bash
python setup.py build install
```

### On Windows platforms with Microsoft Visual Studio

```cmd
python setup.py build -c msvc install
```
