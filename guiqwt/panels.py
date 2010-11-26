# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
guiqwt.panels
-------------

The `panels` module provides :py:class:`guiqwt.curve.PanelWidget` (the `panel` 
widget class from which all panels must derived) and identifiers for each kind 
of panel:
    * :py:data:`guiqwt.panels.ID_ITEMLIST`: ID of the `item list` panel
    * :py:data:`guiqwt.panels.ID_CONTRAST`: ID of the `contrast 
      adjustment` panel
    * :py:data:`guiqwt.panels.ID_XCS`: ID of the `X-axis cross section` 
      panel
    * :py:data:`guiqwt.panels.ID_YCS`: ID of the `Y-axis cross section` 
      panel

.. seealso::
        
    Module :py:mod:`guiqwt.plot`
        Module providing ready-to-use curve and image plotting widgets and 
        dialog boxes
    
    Module :py:mod:`guiqwt.curve`
        Module providing curve-related plot items and plotting widgets
        
    Module :py:mod:`guiqwt.image`
        Module providing image-related plot items and plotting widgets
        
    Module :py:mod:`guiqwt.tools`
        Module providing the `plot tools`
"""

from PyQt4.QtGui import QWidget

# Local imports
from guiqwt.signals import SIG_VISIBILITY_CHANGED


#===============================================================================
# Panel IDs
#===============================================================================

# Item list panel
ID_ITEMLIST = "itemlist"

# Contrast adjustment panel
ID_CONTRAST = "contrast"

# X-cross section panel
ID_XCS = "x_cross_section"

# Y-cross section panel
ID_YCS = "y_cross_section"


#===============================================================================
# Base Panel Widget class
#===============================================================================
class PanelWidget(QWidget):
    PANEL_ID = None # string
    
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        assert self.PANEL_ID is not None
    
    def showEvent(self, event):
        QWidget.showEvent(self, event)
        self.emit(SIG_VISIBILITY_CHANGED, True)
        
    def hideEvent(self, event):
        QWidget.hideEvent(self, event)
        self.emit(SIG_VISIBILITY_CHANGED, False)
