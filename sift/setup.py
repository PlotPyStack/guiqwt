#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Setup script for distributing SIFT as a stand-alone executable
# SIFT is the Signal and Image Filtering Tool
# Simple signal and image processing application based on guiqwt and guidata
# (see guiqwt/sift.pyw)
#
# Buiding instructions:
# python setup_sift.py py2exe

from distutils.core import setup
import py2exe # Patching distutils setup
from guidata.disthelpers import (remove_build_dist, get_default_excludes,
                         get_default_dll_excludes, create_vs2008_data_files,
                         add_modules, add_module_data_files)

from guiqwt import sift
DIST_DIR = "sift"+sift.VERSION.replace('.', '')

# Removing old build/dist folders
remove_build_dist(DIST_DIR)

# Including/excluding DLLs and Python modules
EXCLUDES = get_default_excludes()
INCLUDES = []
DLL_EXCLUDES = get_default_dll_excludes()
DATA_FILES = create_vs2008_data_files()

# Configuring/including Python modules
add_modules(('PyQt4', 'guidata', 'guiqwt'), DATA_FILES, INCLUDES, EXCLUDES)

try:
    import spyderlib
    # Distributing application-specific data files
    add_module_data_files("spyderlib", ("images", ),
                          ('.png', '.svg',), DATA_FILES, copy_to_root=False)
    add_module_data_files("spyderlib", ("", ),
                          ('.mo', '.py'), DATA_FILES, copy_to_root=False)
except ImportError:
    pass

EXCLUDES += ['IPython']

setup(
      options={
               "py2exe": {"compressed": 2, "optimize": 2,
                          "includes": INCLUDES, "excludes": EXCLUDES,
                          "dll_excludes": DLL_EXCLUDES,
                          "dist_dir": DIST_DIR,},
               },
      data_files=DATA_FILES,
      windows=[{
                "script": "../guiqwt/sift.pyw",
                "icon_resources": [(0, "sift.ico")],
                "dest_base": "sift",
                "version": sift.VERSION,
                "company_name": u"Commissariat à l'Energie Atomique et aux Energies Alternatives",
                "copyright": u"Copyright © 2010 CEA - Pierre Raybaut",
                "name": "Sift",
                "description": "Signal and Image Filtering Tool",
                },],
#      zipfile = None,
      )
