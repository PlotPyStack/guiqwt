# -*- coding: utf-8 -*-
#
# Copyright © 2009-2010 CEA
# Pierre Raybaut
# Licensed under the terms of the CECILL License
# (see guiqwt/__init__.py for details)

"""Compile new translations for this module
must be run from toplevel directory !
"""
from guidata.gettext_helpers import do_compile

if __name__ == "__main__":
    do_compile("guiqwt")
