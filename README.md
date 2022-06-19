## Handy image utils
```
sudo apt install poppler-utils pngcheck imagemagick
```

### PDF to PNGs
```
mkdir poulepngs
pdftoppm ~/Downloads/poules-ek-2020_1.pdf ./poulepngs/poule -png -r 300
```

### Add alpha channel to pngs using imagemagick
```
mkdir with-alpha
convert poulepngs/*.png -set filename:fn '%[basename]' 'png32:./with-alpha/%[filename:fn].png'
```
Image magick appears to have some sort of bug, or safeguard that it doesn't process too many images.  
In which case run the following, you might need to `unalias ls`
```
ls poulepngs/*.png | xargs -P 8 -I {} convert {} -set filename:fn '%[basename]' 'png32:./with-alpha/%[filename:fn].png'
```



## TODO
- [ ] dedupe rects
