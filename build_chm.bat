sphinx-build -b htmlhelp doc doctmp
"C:\Program Files\HTML Help Workshop\hhc.exe" doctmp\guiqwtdoc.hhp
copy doctmp\guiqwtdoc.chm .
rmdir /S /Q doctmp