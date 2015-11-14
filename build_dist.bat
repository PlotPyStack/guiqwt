del MANIFEST
rmdir /S /Q build
rmdir /S /Q dist
python setup.py build sdist
python setup.py --no-user-cfg build -c msvc bdist_wininst --sse2
python setup.py --no-user-cfg build -c msvc bdist_wheel --sse2
python setup.py --no-user-cfg build -c msvc bdist_wininst --plat-name=win32 --sse2
python setup.py --no-user-cfg build -c msvc bdist_wheel --plat-name=win32 --sse2