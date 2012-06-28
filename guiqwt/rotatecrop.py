# -*- coding: utf-8 -*-
#
# Copyright © 2012 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)


from PyQt4.QtGui import QCheckBox, QDialog
from PyQt4.QtCore import SIGNAL

from guidata.qthelpers import create_toolbutton
from guidata.configtools import get_icon

# Local imports
from guiqwt.config import _
from guiqwt.builder import make
from guiqwt.plot import ImageDialog
from guiqwt.histogram import lut_range_threshold
from guiqwt.image import INTERP_LINEAR, get_image_in_shape


class RotateCropDialog(ImageDialog):
    """Rotate & Crop Dialog
    
    Rotate and crop a :py:class:`guiqwt.image.TrImageItem` plot item"""
    def __init__(self, parent, wintitle=None, options=None, resize_to=None):
        self.crop_rect = None
        self.item = None
        self.item_original_state = None
        self.item_original_crop = None
        self.item_original_transform = None
        if wintitle is None:
            wintitle = _(u"Rotate & Crop")
        ImageDialog.__init__(self, wintitle=wintitle, edit=True,
                             toolbar=False, options=options, parent=parent)
        if resize_to is not None:
            width, height = resize_to
            self.resize(width, height)
        self.cropped_array = None

    def set_item(self, item):
        self.item = item
        self.item_original_state = (item.can_select(),
                                    item.can_move(),
                                    item.can_resize(),
                                    item.can_rotate())
        self.item_original_crop = item.get_crop()
        self.item_original_transform = item.get_transform()

        self.item.set_selectable(True)
        self.item.set_movable(True)
        self.item.set_resizable(False)
        self.item.set_rotatable(True)
        
        plot = self.get_plot()
        item.set_lut_range(lut_range_threshold(item, 256, 2.))
        item.set_interpolation(INTERP_LINEAR)
        plot.add_item(self.item)
        
        # Setting the item as active item (even if the cropping rectangle item
        # will also be set as active item just below), for the image tools to
        # register this item (contrast, ...):
        plot.set_active_item(self.item)
        self.item.unselect()
        
        self.crop_rect = make.annotated_rectangle(0, 0, 1, 1,
                                                  _(u"Cropping rectangle"))
        self.crop_rect.annotationparam.format = "%.1f cm"
        plot.add_item(self.crop_rect)
        plot.set_active_item(self.crop_rect)
        x0, y0, x1, y1 = self.item.get_crop_coordinates()
        self.crop_rect.set_rect(x0, y0, x1, y1)
        plot.replot()
        
    def get_item(self):
        return self.item
        
    def reset(self):
        self.item.set_crop(*self.item_original_crop)
        self.item.set_transform(*self.item_original_transform)
        x0, y0, x1, y1 = self.item.border_rect.get_rect()
        self.crop_rect.set_rect(x0, y0, x1, y1)
        self.crop()
        self.get_plot().replot()
        
    def install_button_layout(self):
        """Reimplemented ImageDialog method"""
        # Show crop rectangle checkbox
        show_crop = QCheckBox(_(u"Show cropping rectangle"), self)
        show_crop.setChecked(True)
        self.connect(show_crop, SIGNAL("toggled(bool)"), self.show_crop_rect)
        self.button_layout.addWidget(show_crop)
        
        # Advanced options
        edit_options_btn = create_toolbutton(self, text=_(u"Reset"),
                                             icon=get_icon("eraser.png"),
                                             triggered=self.reset,
                                             autoraise=False)
        self.button_layout.addSpacing(15)
        self.button_layout.addWidget(edit_options_btn)
        
        # Apply button
        apply_btn = create_toolbutton(self, text=_(u"Apply"),
                                      icon=get_icon("apply.png"),
                                      triggered=self.apply, autoraise=False)
        self.button_layout.addStretch()
        self.button_layout.addWidget(apply_btn)
        self.button_layout.addStretch()
        
        super(RotateCropDialog, self).install_button_layout()
    
    def show_crop_rect(self, state):
        self.crop_rect.setVisible(state)
        self.crop_rect.label.setVisible(state)
        plot = self.get_plot()
        plot.replot()
        
    def crop(self):
        """Let's crop!"""
        i_points = self.item.border_rect.get_points()
        xmin, ymin = i_points.min(axis=0)
        xmax, ymax = i_points.max(axis=0)
        xc0, yc0, xc1, yc1 = self.crop_rect.shape.get_rect()
        left = max([0, xc0-xmin])
        right = max([0, xmax-xc1])
        top = max([0, yc0-ymin])
        bottom = max([0, ymax-yc1])
        self.item.set_crop(left, top, right, bottom)
#        print "set_crop:", left, top, right, bottom
        self.item.compute_bounds()
        
    def apply(self):
        self.crop()
        self.get_plot().replot()

    def __restore_original_state(self):
        select, move, resize, rotate = self.item_original_state
        self.item.set_selectable(select)
        self.item.set_movable(move)
        self.item.set_resizable(resize)
        self.item.set_rotatable(rotate)

    def accept(self):
        """Reimplement Qt method"""
        self.__restore_original_state()
        self.cropped_array = get_image_in_shape(self.crop_rect,
                                                apply_interpolation=False)
        self.crop()
        # Ignoring image position changes
        pos_x0, pos_y0, _angle, sx, sy, hf, vf = self.item_original_transform
        _pos_x0, _pos_y0, angle, _sx, _sy, hf, vf = self.item.get_transform()
        self.item.set_transform(pos_x0, pos_y0, angle, sx, sy, hf, vf)
        QDialog.accept(self)

    def reject(self):
        """Reimplement Qt method"""
        self.__restore_original_state()
        self.item.set_crop(*self.item_original_crop)
        self.item.set_transform(*self.item_original_transform)
        QDialog.reject(self)
