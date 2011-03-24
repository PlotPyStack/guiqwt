Making executable Windows programs
==================================

The `py2exe` Python library is an extension to Python `distutils` module which 
converts Python scripts into executable Windows programs, able to run without 
requiring a Python installation.

Making such an executable program may be a non trivial task when the script 
dependencies include libraries with data or extensions, such as `PyQt4` or 
`guidata` and `guiqwt`. This task has been considerably simplified thanks to 
the helper functions provided by :py:mod:`guidata.disthelpers`.

Example
~~~~~~~

This example is included in `guiqwt` source package (see the 
``py2exe_example`` directory).

Simple example script named ``simpledialog.pyw`` which is based on `guiqwt` 
(and implicitely on `guidata`)::

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
        qapplication()
        dlg = VerySimpleDialog()
        dlg.set_data(np.random.rand(100, 100))
        dlg.exec_()

The ``setup.py`` script may be written as the following::

    from distutils.core import setup
    import py2exe # Patching distutils setup
    from guidata.disthelpers import (remove_build_dist, get_default_excludes,
                             get_default_dll_excludes, create_vs2008_data_files,
                             add_modules)
    
    # Removing old build/dist folders
    remove_build_dist()
    
    # Including/excluding DLLs and Python modules
    EXCLUDES = get_default_excludes()
    INCLUDES = []
    DLL_EXCLUDES = get_default_dll_excludes()
    DATA_FILES = create_vs2008_data_files()
    
    # Configuring/including Python modules
    add_modules(('PyQt4', 'guidata', 'guiqwt'), DATA_FILES, INCLUDES, EXCLUDES)
    
    setup(
          options={
                   "py2exe": {"compressed": 2, "optimize": 2, 'bundle_files': 1,
                              "includes": INCLUDES, "excludes": EXCLUDES,
                              "dll_excludes": DLL_EXCLUDES,
                              "dist_dir": "dist",},
                   },
          data_files=DATA_FILES,
          windows=[{
                    "script": "simpledialog.pyw",
                    "dest_base": "simpledialog",
                    "version": "1.0.0",
                    "company_name": u"CEA",
                    "copyright": u"Copyright © 2010 CEA - Pierre Raybaut",
                    "name": "Simple dialog box",
                    "description": "Simple dialog box",
                    },],
          zipfile = None,
          )

Make the Windows executable program with the following command:
    `python setup.py py2exe`
