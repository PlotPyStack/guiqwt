# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""
Base classes for guiqwt module
"""

from PyQt4.QtGui import QWidget

# Local imports
from guiqwt.signals import SIG_VISIBILITY_CHANGED


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
