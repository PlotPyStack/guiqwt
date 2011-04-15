# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Very simple dialog box"""

from guiqwt.plot import ImageDialog
from guiqwt.builder import make

class VerySimpleDialog(ImageDialog):
    def set_data(self, data):
        plot = self.get_plot()
        item = make.trimage(data)
        plot.add_item(item, z=0)
        plot.set_active_item(item)
        plot.replot()

if __name__ == "__main__":
    import numpy as np
    from guidata import qapplication
    _app = qapplication()
    dlg = VerySimpleDialog()
    dlg.set_data(np.random.rand(100, 100))
    dlg.exec_() # No need to call app.exec_: a dialog box has its own event loop
