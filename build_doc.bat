rmdir /S /Q build\doc
rmdir /S /Q build\doc_chm
sphinx-build -b html doc build\doc
sphinx-build -b htmlhelp doc build\doc_chm
pause