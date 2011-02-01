# -*- coding: utf-8 -*-
#
# Copyright © 2009-2011 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.transitional
-------------------

The purpose of this transitional package is to regroup all the references to 
the ``PyQwt`` library.

No other ``guiqwt`` module should import ``PyQwt`` or use any of its 
interfaces directly.
"""

from PyQt4.Qwt5 import (QwtPlot, QwtSymbol, QwtLinearScaleEngine,
                        QwtLog10ScaleEngine, QwtText, QwtPlotCanvas,
                        QwtLinearColorMap, QwtDoubleInterval, toQImage,
                        QwtPlotCurve, QwtPlotGrid, QwtPlotItem, QwtScaleMap,
                        QwtPlotMarker, QwtPlotPrintFilter)

