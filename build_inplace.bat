rmdir /S /Q build
python setup.py build_ext -c mingw32 --inplace
pause