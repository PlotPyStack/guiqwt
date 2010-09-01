del MANIFEST
rmdir /S /Q build
rmdir /S /Q dist
python setup.py build -c mingw32 bdist_wininst
python setup.py build sdist
pause