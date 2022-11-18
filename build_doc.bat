sphinx-build -b htmlhelp doc doctmp
"C:\Program Files\HTML Help Workshop\hhc.exe" doctmp\guiqwt.hhp
"C:\Program Files (x86)\HTML Help Workshop\hhc.exe" doctmp\guiqwt.hhp
copy doctmp\guiqwt.chm .
"C:\Program Files\7-Zip\7z.exe" a guiqwt.chm.zip guiqwt.chm
del doctmp\guiqwt.chm
del doc.zip
del doctmp\guiqwt.hh*
sphinx-build -b html doc doctmp
cd doctmp
"C:\Program Files\7-Zip\7z.exe" a -r ..\doc.zip *.*
cd ..
rmdir /S /Q doctmp
del guiqwt.chm.zip