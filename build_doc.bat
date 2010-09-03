rmdir /S /Q spyderlib\doc
sphinx-build -b html doc_src build\doc
sphinx-build -b htmlhelp doc_src build\doc_chm
pause