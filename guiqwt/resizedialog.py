# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

from __future__ import division

from PyQt4.QtGui import (QDialog, QDialogButtonBox, QVBoxLayout, QFormLayout,
                         QLineEdit, QIntValidator, QLabel)
from PyQt4.QtCore import SIGNAL, SLOT, Qt


from guiqwt.config import _


def is_edit_valid(edit):
    text = edit.text()
    state, _t = edit.validator().validate(text, 0)
    return state == QIntValidator.Acceptable


class ResizeDialog(QDialog):
    def __init__(self, parent, new_size, old_size, text=""):
        QDialog.__init__(self, parent)
        
        self.width, self.height = new_size
        self.old_width, self.old_height = old_size
        self.ratio = self.width/self.height

        layout = QVBoxLayout()
        self.setLayout(layout)
        
        formlayout = QFormLayout()
        layout.addLayout(formlayout)
        
        if text:
            label = QLabel(text)
            label.setAlignment(Qt.AlignHCenter)
            formlayout.addRow(label)
        
        self.w_edit = w_edit = QLineEdit(self)
        w_valid = QIntValidator(w_edit)
        w_valid.setBottom(1)
        w_edit.setValidator(w_valid)
                     
        self.h_edit = h_edit = QLineEdit(self)
        h_valid = QIntValidator(h_edit)
        h_valid.setBottom(1)
        h_edit.setValidator(h_valid)

        formlayout.addRow(_("Width (pixels)"), w_edit)
        formlayout.addRow(_("Height (pixels)"), h_edit)
        
        formlayout.addRow(_("Original size:"), QLabel("%d x %d" % old_size))
        self.z_label = QLabel()
        formlayout.addRow(_("Zoom factor:"), self.z_label)
        
        # Button box
        self.bbox = bbox = QDialogButtonBox(QDialogButtonBox.Ok |
                                            QDialogButtonBox.Cancel)
        self.connect(bbox, SIGNAL("accepted()"), SLOT("accept()"))
        self.connect(bbox, SIGNAL("rejected()"), SLOT("reject()"))
        layout.addWidget(bbox)

        self.w_edit.setText(str(self.width))
        self.h_edit.setText(str(self.height))
        self.update_widgets()
        
        self.setWindowTitle(_("Resize"))
        
        self.connect(w_edit, SIGNAL("textChanged(QString)"),
                     self.width_changed)
        self.connect(h_edit, SIGNAL("textChanged(QString)"),
                     self.height_changed)

    def update_widgets(self):
        valid = True
        for edit in (self.w_edit, self.h_edit):
            if not is_edit_valid(edit):
                valid = False
        self.bbox.button(QDialogButtonBox.Ok).setEnabled(valid)
        self.z_label.setText("%d %s" % (100*self.width/self.old_width, '%'))
        
    def width_changed(self, text):
        if is_edit_valid(self.sender()):
            self.width = int(text)
            self.height = int(self.width/self.ratio)
            self.h_edit.blockSignals(True)
            self.h_edit.setText(str(self.height))
            self.h_edit.blockSignals(False)
        self.update_widgets()

    def height_changed(self, text):
        if is_edit_valid(self.sender()):
            self.height = int(text)
            self.width = int(self.ratio*self.height)
            self.w_edit.blockSignals(True)
            self.w_edit.setText(str(self.width))
            self.w_edit.blockSignals(False)
        self.update_widgets()

    
if __name__ == '__main__':
    import guidata
    qapp = guidata.qapplication()
    test = ResizeDialog(None, (150, 100), (300, 200), "Enter the new size:")
    if test.exec_():
        print test.width
        print test.get_scale_factor()
