Installation
============

Dependencies
------------

Requirements:
    * Python 2.x (x>=6) or 3.x (x>=2)
    * PyQt4 4.x (x>=3 ; recommended x>=4) or PyQt5 5.x (x>=5)
    * PythonQwt >=0.5
    * guidata >=1.7
    * NumPy
    * SciPy
    * Pillow
    
Optional Python modules:
    * spyderlib 2.1 for Sift embedded Python console
    * pydicom >=0.9.3 for DICOM files I/O features

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
