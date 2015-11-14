rmdir /S /Q build
del guiqwt\*.pyd
python setup.py --no-user-cfg build_ext -c msvc --inplace --sse2