del MANIFEST
python setup.py build sdist
python setup.py --no-user-cfg build -c msvc bdist_wheel --sse2
python setup.py --no-user-cfg build -c msvc bdist_wheel --plat-name=win32 --sse2
