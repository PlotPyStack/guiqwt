Migrating from version 2 to version 4
=====================================

The main change between version 2 and version 4 is the basic plotting library
on which `guiqwt` is based on:

  * `guiqwt` version 2: depends on `PyQwt`, the Python bindings to `Qwt` C++
    library -- only supports PyQt4.

  * `guiqwt` version 4: depends on `PythonQwt`, a new library written from
    scratch to continue supporting `Qwt` API through a pure Python
    reimplementation of its main classes (`QwtPlot`, `QwtPlotItem`,
    `QwtPlotCanvas`, ...) -- supports PyQt5 and PySide2.

Another major change is the switch from old-style to new-style signals and
slots. The :py:mod:`guiqwt.signals` module is now empty because it used to
collect strings for old-style signals: however, it still contains
documentation on available signals.

Examples
~~~~~~~~

Switching from `PyQwt` to `PythonQwt` in your code::

    from PyQt4.Qwt5 import QwtPlot  # PyQwt (supports only PyQt4)

    from qwt import QwtPlot  # PythonQwt (supports PyQt5 and PySide2)

Switching from `guiqwt 2` to `guiqwt 4`::

    plot = get_plot_instance()  # plot is a QwtPlot instance

    ## guiqwt 2:
    from guiqwt.signals import SIG_ITEM_MOVED
    plot.connect(plot, SIG_ITEM_MOVED, item_was_moved)

    ## guiqwt 4:
    plot.SIG_ITEM_MOVED.connect(item_was_moved)
