del MANIFEST
rmdir /S /Q build
rmdir /S /Q dist
python setup.py build -c mingw32 bdist_wininst --sse2
python setup.py build sdist
pause