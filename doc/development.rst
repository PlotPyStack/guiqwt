How to contribute
=================

Submitting changes
------------------

Due to confidentiality issues, we are not able *for now* to publish any source-
controlled repository (even if we do have a `Mercurial` repository for the 
project). However, this does not prevent motivated users from contributing to 
the project by sending patches applied to the last published version of the 
library. To compensate the absence of source repository, we try to update the 
library as often as we can in order to keep the public source archive version 
as close as possible to the internal development version.

Coding guidelines
-----------------

In general, we try to follow the standard Python coding guidelines, which cover 
all the important coding aspects (docstrings, comments, naming conventions, 
import statements, ...) as described here:

* `Style Guide for Python Code  <http://www.python.org/peps/pep-0008.html>`_  

The easiest way to check that your code is following those guidelines is to 
run `pylint` (a note greater than 8/10 seems to be a reasonnable goal).

PyQt v4.4 compatibility issues
------------------------------

The project has to be compatible with PyQt >= v4.4 which means that the 
following recommendations should be followed:

* avoid using `super`: when writing classes deriving from a QObject child class 
  (i.e. almost any single class imported from QtGui or QtCore), the `super` 
  builtin-function should not be used outside the constructor method (call 
  the parent class method directly instead)

* before using any function or method from PyQt4, please check that the feature 
  you are about to use was already implemented in PyQt4 v4.4 (more precisely 
  in the Qt version used in PyQt4 v4.4) -- if not, a workaround should be 
  implemented to avoid breaking compatibility

PyQt / PySide compatibility
---------------------------

In the near future, the project will be officially compatible with both PyQt 
and PySide.

In its current implementation, it has to be compatible with PyQt API #1 (old 
PyQt versions) and API #2 (PySide-compatible API, PyQt >= v4.6), which means 
that the following recommendations should be followed:

* `QVariant` objects must not be used (API #2 compatibility)

* `QString` and `QStringList` objects must not be used (API #2 compatibility)

* When connecting built-in C++ signals which were originally made to pass 
  strings (or string lists), the arguments should always be assumed to be 
  `QString` (or `QStringList`) objects (API #1 compatibility) and so be 
  converted systematically to the Python equivalent object, i.e. unicode 
  (or list).
