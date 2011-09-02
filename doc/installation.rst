Installation
============

Dependencies
------------

Requirements:
    * Python 2.x (x>=5)
    * PyQt4 4.x (x>=3 ; recommended x>=4)
    * PyQwt 5.x (x>=2)
    * guidata 1.3.2 or later
    * NumPy 1.x (x>=3) -- NumPy 1.6 or later is required for Windows binaries
    * SciPy 0.x (x>=7)
    * PIL 1.1.x (x>=6)
    
Optional Python modules:
    * spyderlib 2.1 for Sift embedded Python console
    * pydicom 0.9.x (x>=4) for DICOM files I/O features

Installation
------------

All platforms:

    The ``setup.py`` script supports the following extra options for 
    optimizing the image scaler engine with SSE2/SSE3 processors:
    ``--sse2`` and ``--sse3``

On GNU/Linux and MacOS platforms:
    ``python setup.py build install``

    If `gfortran` is not your default Fortran compiler:
	``python setup.py build --fcompiler=gfortran install``
    or if it fails, you may try the following:
	``python setup.py build_ext --fcompiler=gnu95 build install``
    
On Windows platforms (requires MinGW with gfortran):
    ``python setup.py build -c mingw32 install``

Help and support
----------------

External resources:
    * Bug reports and feature requests: `GoogleCode`_
    * Help, support and discussions around the project: `GoogleGroup`_

.. _GoogleCode: http://guiqwt.googlecode.com
.. _GoogleGroup: http://groups.google.fr/group/guidata_guiqwt
