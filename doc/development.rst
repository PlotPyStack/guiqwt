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

* do not use the PyQt-specific QFileDialog static methods (not present in Qt) 
  which were introduced in PyQt v4.6: `getOpenFileNameAndFilter`, 
  `getOpenFileNamesAndFilter` and `getSaveFileNameAndFilter` (`guidata` 
  provides wrappers around `QFileDialog` static methods handling the selected 
  filter which were taken from the `spyderlib` library (from module 
  `spyderlib.qt.compat`): they are available in `guidata.qt.compat`)

PyQt / PySide compatibility
---------------------------

The project should be mostly compatible with both PyQt and PySide (although 
PySide is not as popular as it used to be, so testing tend to be limited).

PyQt5 compatibility
-------------------

In its current implementation, the code base has to be compatible with PyQt 
API #2 (PySide-compatible API, PyQt >= v4.6) and with PyQt5, which means that 
the following recommendations should be followed:

* `QVariant` objects must not be used (API #2 compatibility)

* Use exclusively new-style signals and slots

* Read carefully PyQt5 documentation regarding class inheritance behavior: it 
  is quite different than the old PyQt4 implementation. Producing code 
  compatible with both PyQt4 and PyQt5 can be tricky: testing is essential.


Python 3 compatibility
======================

Regarding Python 3 compatibility, we chose to handle it by maintaining a single
source branch being compatible with both Python 2.6-2.7 and Python 3.

Here is what we have done.

Fixing trivial things with 2to3
-------------------------------

The first step is to run the `2to3` script (see Python documentation) to 
convert print statements to print function calls -- note that your source 
directory (named `directory_name`) has to be version controlled (no backup is 
done thanks to the `-n` option flag).
`python 2to3.py -w -n -f print directory_name`

Open each modified source file and add the following line at the beginning:
from __future__ import print_function

Then run again `2to3` with all other Python 2/3 compatible fixers:
`python 2to3.py -w -n -f apply -f dict -f except -f exitfunc -f filter -f has_key -f map -f ne -f raise -f ws_comma -f xrange -f xreadlines -f zip directory_name`

After these two steps, your code should be compatible with Python 2.6, 2.7 
and 3.x, but only with respect to the simplest changes that occured between 
Python 2 and Python 3. However, this a step forward to Python 3 compatibility 
without breaking Python 2.6+ compatibility.

Fixing unicode issues
---------------------

In Python 3, `unicode` and `str` strings have been replaced by `str` and 
`bytes` strings:

  * `str` is the text string type, supporting unicode characters natively

  * `bytes` is the binary string type.

As a consequence, Python 2 code involving strings may cause compatibility 
issues with Python 3. For example:

  * file I/O may return `bytes` instead of `str` in Python 3 (depending on the 
    open mode): this can be solved by calling the `decode` method on the `bytes` 
    object (this will work on both Python 2 `str` and Python 3 `bytes` objects)

  * in Python 3.0-3.2, the `u'unicode text'` or `u"unicode text"` syntax is 
    not allowed and will raise a SyntaxError: this can be solved by inserting the 
    `from __future__ import unicode_literals` at the beginning of the script and 
    by removing all the `u` string prefixes

  * in Python 3 `isinstance(text, basestring)` can be replaced by 
    `is_text_string(text)` (function of the `guidata.py3compat` module)

  * in Python 3 `isinstance(text, unicode)` can be replaced by 
    `is_unicode(text)` (function of the `guidata.py3compat` module)

  * in Python 3 `unicode(text)` can be replaced by `to_text_string(text)` 
    (function of the `guidata.py3compat` module)
