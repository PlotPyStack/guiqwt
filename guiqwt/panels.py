# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
Panels for Plot Manager (see guiqwt.plot.PlotManager class)
"""

from PyQt4.QtGui import QWidget

# Local imports
from guiqwt.signals import SIG_VISIBILITY_CHANGED


#===============================================================================
# Panel IDs
#===============================================================================

# Item list panel
ITEMLIST_PANEL_ID = "itemlist"

# Contrast adjustment panel
CONTRAST_PANEL_ID = "contrast"

# X-cross section panel
XCS_PANEL_ID = "x_cross_section"

# Y-cross section panel
YCS_PANEL_ID = "y_cross_section"


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
