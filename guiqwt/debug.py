# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.debug
------------

The `debug` module contains some debugging functions (mostly dumping attributes
of Qt Objects).
"""

from StringIO import StringIO
from PyQt4.QtGui import QImage, QInputEvent
from PyQt4.QtCore import Qt, QEvent

def buttons_to_str(buttons):
    """Conversion des flags Qt en chaine"""
    s = ""
    if buttons & Qt.LeftButton:
        s += "L"
    if buttons & Qt.MidButton:
        s += "M"
    if buttons & Qt.RightButton:
        s += "R"
    return s


def evt_type_to_str( type ):
    """Représentation textuelle d'un type d'événement (debug)"""
    if type == QEvent.MouseButtonPress:
        return "Mpress"
    elif type == QEvent.MouseButtonRelease:
        return "Mrelease"
    elif type == QEvent.MouseMove:
        return "Mmove"
    elif type == QEvent.ContextMenu:
        return "Context"
    else:
        return "%d" % type
    
    
def print_event(evt):
    """Représentation textuelle d'un événement (debug)"""
    s = ""
    if isinstance(evt, QInputEvent):
        s += evt_type_to_str( evt.type() )
        s += "%08x:" % evt.modifiers()
        if hasattr(evt, "buttons"):
            buttons = evt.buttons()
        elif hasattr(evt, "buttons"):
            buttons = evt.button()
        else:
            buttons = 0
        s += buttons_to_str(buttons)
    if s:
        print s
    else:
        print evt


def qimage_format( fmt ):
    for attr in dir(QImage):
        if attr.startswith("Format"):
            val = getattr(QImage, attr)
            if val == fmt:
                return attr[len("Format_"):]
    return str(fmt)
    
def qimage_to_str( img, indent="" ):
    s = StringIO()
    print >>s, indent, img
    indent += "  "
    print >>s, indent, "Size:", img.width(), "x", img.height()
    print >>s, indent, "Depth:", img.depth()
    print >>s, indent, "Format", qimage_format(img.format())
    return s.getvalue()
