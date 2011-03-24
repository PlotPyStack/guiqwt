#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Buiding instructions:
# python setup.py py2exe

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
