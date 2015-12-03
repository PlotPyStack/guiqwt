del MANIFEST
python setup.py build sdist upload
python setup.py --no-user-cfg build -c msvc bdist_wininst --sse2 upload
python setup.py --no-user-cfg build -c msvc bdist_wheel --sse2 upload
python setup.py --no-user-cfg build -c msvc bdist_wininst --plat-name=win32 --sse2 upload
python setup.py --no-user-cfg build -c msvc bdist_wheel --plat-name=win32 --sse2 upload
pause
