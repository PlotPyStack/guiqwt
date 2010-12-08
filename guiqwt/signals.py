# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.signals
--------------

The `signals` module contains constants defining the custom Qt SIGNAL objects 
used by `guiqwt`: the signals definition are gathered here to avoid mispelling
signals at connect and emit sites.

Signals available:
    :py:data:`guiqwt.signals.SIG_ITEM_MOVED`
        Emitted by IBasePlotItem object. Object was moved (args: x0, y0, x1, y1)
    :py:data:`guiqwt.signals.SIG_MARKER_CHANGED`
        Emitted by shapes.Marker when the position changes
    :py:data:`guiqwt.signals.SIG_AXES_CHANGED`
        Emitted by shapes.Axes when the position (or the angle) changes
    :py:data:`guiqwt.signals.SIG_ANNOTATION_CHANGED`
        Emitted by annotation.AnnotatedShape when the position changes
    :py:data:`guiqwt.signals.SIG_RANGE_CHANGED`
        Emitted by shapes.XRangeSelection when the range changes
    :py:data:`guiqwt.signals.SIG_ITEMS_CHANGED`
        Emitted by plot when item list has changed (item removed, added, ...)
    :py:data:`guiqwt.signals.SIG_ACTIVE_ITEM_CHANGED`
        Emitted by plot when selected item has changed
    :py:data:`guiqwt.signals.SIG_ITEM_REMOVED`
        Emitted by "itemlist" panel when an item was deleted from the list
    :py:data:`guiqwt.signals.SIG_ITEM_SELECTION_CHANGED`
        Emitted by plot when an item is selected
    :py:data:`guiqwt.signals.SIG_PLOT_LABELS_CHANGED`
        Emitted (by plot) when plot's title or any axis label has changed
    :py:data:`guiqwt.signals.SIG_AXIS_DIRECTION_CHANGED`
        Emitted (by plot) when any plot axis direction has changed
    :py:data:`guiqwt.signals.SIG_VOI_CHANGED`
        Emitted by "contrast" panel's histogram when the lut range of some items
        changed (for now, this signal is for guiqwt.histogram module's internal 
        use only - the 'public' counterpart of this signal is SIG_LUT_CHANGED, 
        see below)
    :py:data:`guiqwt.signals.SIG_LUT_CHANGED`
        Emitted by plot when LUT has been changed by the user
    :py:data:`guiqwt.signals.SIG_VISIBILITY_CHANGED`
        Emitted for example by panels when their visibility has changed
    :py:data:`guiqwt.signals.SIG_VALIDATE_TOOL`
        Emitted by an interactive tool to notify that the tool has just been 
        "validated", i.e. <ENTER>, <RETURN> or <SPACE> was pressed
"""
from PyQt4.QtCore import SIGNAL

# Emitted by IBasePlotItem object - object was moved (args: x0, y0, x1, y1)
SIG_ITEM_MOVED = SIGNAL("item_moved(PyQt_PyObject,double,double,double,double)")

# Emitted by shapes.Marker when the position changes
SIG_MARKER_CHANGED = SIGNAL("marker_changed(PyQt_PyObject)")

# Emitted by shapes.Axes when the position (or the angle) changes
SIG_AXES_CHANGED = SIGNAL("axes_changed(PyQt_PyObject)")

# Emitted by annotation.AnnotatedShape when the position changes
SIG_ANNOTATION_CHANGED = SIGNAL("annotation_changed(PyQt_PyObject)")

# Emitted by shapes.XRangeSelection when the range changes
SIG_RANGE_CHANGED = SIGNAL("range_changed(PyQt_PyObject,double,double)")

# Emitted by plot when item list has changed (item removed, added, ...)
SIG_ITEMS_CHANGED = SIGNAL('items_changed(PyQt_PyObject)')

# Emitted by plot when selected item has changed
SIG_ACTIVE_ITEM_CHANGED = SIGNAL('active_item_changed(PyQt_PyObject)')

# Emitted by "itemlist" panel when an item was deleted from the list
SIG_ITEM_REMOVED = SIGNAL('item_removed(PyQt_PyObject)')

# Emitted by plot when an item is selected
SIG_ITEM_SELECTION_CHANGED = SIGNAL('item_selection_changed(PyQt_PyObject)')

# Emitted (by plot) when plot's title or any axis label has changed
SIG_PLOT_LABELS_CHANGED = SIGNAL('plot_labels_changed(PyQt_PyObject)')

# Emitted (by plot) when any plot axis direction has changed
SIG_AXIS_DIRECTION_CHANGED = SIGNAL('axis_direction_changed(PyQt_PyObject,PyQt_PyObject)')

# Emitted by "contrast" panel's histogram when the lut range of 
# some items changed (for now, this signal is for guiqwt.histogram module's
# internal use only - the 'public' counterpart of this signal 
# is SIG_LUT_CHANGED, see below)
SIG_VOI_CHANGED = SIGNAL("voi_changed()")

# Emitted by plot when LUT has been changed by the user
SIG_LUT_CHANGED = SIGNAL("lut_changed(PyQt_PyObject)")

# Emitted for example by panels when their visibility has changed
SIG_VISIBILITY_CHANGED = SIGNAL("visibility_changed(bool)")

# Emitted by an interactive tool to notify that the tool has just been 
# "validated", i.e. <ENTER>, <RETURN> or <SPACE> was pressed
SIG_VALIDATE_TOOL = SIGNAL("validate_tool")

#===============================================================================
# Event filter related signals (private)
#===============================================================================

SIG_CLICK_EVENT = SIGNAL("click_event")
SIG_START_TRACKING = SIGNAL("start_tracking")
SIG_STOP_NOT_MOVING = SIGNAL("stop_notmoving")
SIG_STOP_MOVING = SIGNAL("stop_moving")
SIG_MOVE = SIGNAL("move")
SIG_END_RECT = SIGNAL("end_rect")
