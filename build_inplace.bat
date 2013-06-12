rmdir /S /Q build
python setup.py build_ext --inplace --sse2
pause