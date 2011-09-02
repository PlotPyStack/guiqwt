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
        Emitted by plot when an IBasePlotItem-like object was moved from 
        (x0, y0) to (x1, y1)
        
        Arguments: item object, x0, y0, x1, y1
    :py:data:`guiqwt.signals.SIG_MARKER_CHANGED`
        Emitted by plot when a `guiqwt.shapes.Marker` position changes
        
        Arguments: `guiqwt.shapes.Marker` object
    :py:data:`guiqwt.signals.SIG_AXES_CHANGED`
        Emitted by plot when a `guiqwt.shapes.Axes` position (or angle) changes
        
        Arguments: `guiqwt.shapes.Axes` object
    :py:data:`guiqwt.signals.SIG_ANNOTATION_CHANGED`
        Emitted by plot when an annotations.AnnotatedShape position changes
        
        Arguments: annotation item
    :py:data:`guiqwt.signals.SIG_RANGE_CHANGED`
        Emitted by plot when a shapes.XRangeSelection range changes
        
        Arguments: range object, lower_bound, upper_bound
    :py:data:`guiqwt.signals.SIG_ITEMS_CHANGED`
        Emitted by plot when item list has changed (item removed, added, ...)
        
        Arguments: plot
    :py:data:`guiqwt.signals.SIG_ACTIVE_ITEM_CHANGED`
        Emitted by plot when selected item has changed
        
        Arguments: plot
    :py:data:`guiqwt.signals.SIG_ITEM_REMOVED`
        Emitted by plot when an item was deleted from the itemlist or using 
        the delete item tool
        
        Arguments: removed item
    :py:data:`guiqwt.signals.SIG_ITEM_SELECTION_CHANGED`
        Emitted by plot when an item is selected

        Arguments: plot
    :py:data:`guiqwt.signals.SIG_PLOT_LABELS_CHANGED`
        Emitted (by plot) when plot's title or any axis label has changed
        
        Arguments: plot
    :py:data:`guiqwt.signals.SIG_AXIS_DIRECTION_CHANGED`
        Emitted (by plot) when any plot axis direction has changed
        
        Arguments: plot
    :py:data:`guiqwt.signals.SIG_VOI_CHANGED`
        Emitted by "contrast" panel's histogram when the lut range of some items
        changed (for now, this signal is for guiqwt.histogram module's internal 
        use only - the 'public' counterpart of this signal is SIG_LUT_CHANGED, 
        see below)
        
    :py:data:`guiqwt.signals.SIG_LUT_CHANGED`
        Emitted by plot when LUT has been changed by the user

        Arguments: plot
    :py:data:`guiqwt.signals.SIG_MASK_CHANGED`
        Emitted by plot when image mask has changed

        Arguments: MaskedImageItem object
    :py:data:`guiqwt.signals.SIG_VISIBILITY_CHANGED`
        Emitted for example by panels when their visibility has changed

        Arguments: state (boolean)
    :py:data:`guiqwt.signals.SIG_VALIDATE_TOOL`
        Emitted by an interactive tool to notify that the tool has just been 
        "validated", i.e. <ENTER>, <RETURN> or <SPACE> was pressed

        Arguments: filter
"""
from guidata.qt.QtCore import SIGNAL

# Emitted by plot when an IBasePlotItem object was moved (args: x0, y0, x1, y1)
SIG_ITEM_MOVED = SIGNAL("item_moved(PyQt_PyObject,double,double,double,double)")

# Emitted by plot when a shapes.Marker position changes
SIG_MARKER_CHANGED = SIGNAL("marker_changed(PyQt_PyObject)")

# Emitted by plot when a shapes.Axes position (or the angle) changes
SIG_AXES_CHANGED = SIGNAL("axes_changed(PyQt_PyObject)")

# Emitted by plot when an annotation.AnnotatedShape position changes
SIG_ANNOTATION_CHANGED = SIGNAL("annotation_changed(PyQt_PyObject)")

# Emitted by plot when the a shapes.XRangeSelection range changes
SIG_RANGE_CHANGED = SIGNAL("range_changed(PyQt_PyObject,double,double)")

# Emitted by plot when item list has changed (item removed, added, ...)
SIG_ITEMS_CHANGED = SIGNAL('items_changed(PyQt_PyObject)')

# Emitted by plot when selected item has changed
SIG_ACTIVE_ITEM_CHANGED = SIGNAL('active_item_changed(PyQt_PyObject)')

# Emitted by plot when an item was deleted from the item list or using the 
# delete item tool
SIG_ITEM_REMOVED = SIGNAL('item_removed(PyQt_PyObject)')

# Emitted by plot when an item is selected
SIG_ITEM_SELECTION_CHANGED = SIGNAL('item_selection_changed(PyQt_PyObject)')

# Emitted by plot when plot's title or any axis label has changed
SIG_PLOT_LABELS_CHANGED = SIGNAL('plot_labels_changed(PyQt_PyObject)')

# Emitted by plot when any plot axis direction has changed
SIG_AXIS_DIRECTION_CHANGED = SIGNAL('axis_direction_changed(PyQt_PyObject,PyQt_PyObject)')

# Emitted by "contrast" panel's histogram when the lut range of 
# some items changed (for now, this signal is for guiqwt.histogram module's
# internal use only - the 'public' counterpart of this signal 
# is SIG_LUT_CHANGED, see below)
SIG_VOI_CHANGED = SIGNAL("voi_changed()")

# Emitted by plot when LUT has been changed by the user
SIG_LUT_CHANGED = SIGNAL("lut_changed(PyQt_PyObject)")

# Emitted by plot when image mask has changed
SIG_MASK_CHANGED = SIGNAL("mask_changed(PyQt_PyObject)")

# Emitted for example by panels when their visibility has changed
SIG_VISIBILITY_CHANGED = SIGNAL("visibility_changed(bool)")

# Emitted by an interactive tool to notify that the tool has just been 
# "validated", i.e. <ENTER>, <RETURN> or <SPACE> was pressed
SIG_VALIDATE_TOOL = SIGNAL("validate_tool")

# Emitted by an interactive tool to notify that it is finished doing its job
SIG_TOOL_JOB_FINISHED = SIGNAL("tool_job_finished")

# Emitted by cross section plot when cross section curve data has changed
SIG_CS_CURVE_CHANGED = SIGNAL("cs_curve_changed(PyQt_PyObject)")

# Emitted by the ImageMaskTool when applying the shape-defined mask
SIG_APPLIED_MASK_TOOL = SIGNAL("applied_mask_tool()")

#===============================================================================
# Event filter related signals (private)
#===============================================================================

SIG_CLICK_EVENT = SIGNAL("click_event")
SIG_START_TRACKING = SIGNAL("start_tracking")
# Emitted when a plot axis' scale changes
SIG_PLOT_AXIS_CHANGED = SIGNAL("plot_axis_changed(PyQt_PyObject)")
SIG_STOP_NOT_MOVING = SIGNAL("stop_notmoving")
SIG_STOP_MOVING = SIGNAL("stop_moving")
SIG_MOVE = SIGNAL("move")
SIG_END_RECT = SIGNAL("end_rect")
