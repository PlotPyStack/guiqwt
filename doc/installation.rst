Installation
============

Dependencies
------------

Requirements:
    * Python 3.x (x>=7)
    * `PyQt5`_ 5.x (x>=5)
    * `PythonQwt`_ >=0.10
    * `guidata`_ >=3.0
    * `QtPy`_ >= 1.3
    * `NumPy`_, `SciPy`_ and `Pillow`_

Optional Python modules:
    * `pydicom`_ >=0.9.3 for DICOM files I/O features

.. _PyQt5: https://pypi.python.org/pypi/PyQt5
.. _PythonQwt: https://pypi.python.org/pypi/PythonQwt
.. _guidata: https://pypi.python.org/pypi/guidata
.. _qtpy: https://pypi.org/project/QtPy/
.. _NumPy: https://pypi.python.org/pypi/NumPy
.. _SciPy: https://pypi.python.org/pypi/SciPy
.. _Pillow: https://pypi.python.org/pypi/Pillow
.. _pydicom: https://pypi.python.org/pypi/pydicom

Installation
------------

All platforms:

    The ``setup.py`` script supports the following extra options for
    optimizing the image scaler engine with SSE2/SSE3 processors:
    ``--sse2`` and ``--sse3``

On GNU/Linux and MacOS platforms:
    ``python setup.py build install``

On Windows platforms with MinGW:
    ``python setup.py build -c mingw32 install``

On Windows platforms with Microsoft Visual C++ compiler:
    ``python setup.py build -c msvc install``

Help and support
----------------

External resources:
    * Bug reports and feature requests: `GitHub`_
    * Help, support and discussions around the project: `GoogleGroup`_

.. _GitHub: https://github.com/PierreRaybaut/guiqwt
.. _GoogleGroup: http://groups.google.fr/group/guidata_guiqwt
