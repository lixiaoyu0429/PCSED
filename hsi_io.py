import spectral
import spectral.io.envi as envi
from pathlib import Path
import numpy as np

def __fixHDRfile(filePath):
    """
    Append the required "byte order" property into the .hdr file to fix the error\n
    修复.hdr文件
    ENVI Header Format: https://www.l3harrisgeospatial.com/docs/enviheaderfiles.html
    """
    
    with open(filePath, encoding='utf-8') as hdrFile:
        hdrInfo = hdrFile.read()

    if not "byte order" in hdrInfo:
        with open(filePath, 'a', encoding='utf-8') as hdrFile:
            hdrFile.write('\nbyte order = 0')

    with open(filePath, encoding='utf-8') as hdrFile:
        hdrInfo = hdrFile.readlines()

    if not "ENVI" in hdrInfo[0] and 'Wayho' in hdrInfo[0]:
        hdrInfo[0] = 'ENVI'
        with open(filePath, 'w', encoding='utf-8') as hdrFile:
            hdrFile.writelines(hdrInfo)
        


def load_envi_img(hdr_path):
    try:
        img = envi.open(hdr_path)
    except:
        __fixHDRfile(hdr_path)
        img = envi.open(hdr_path)
    return img

def save_envi_img(img, original_metadata, out_path):
    H, W, B = img.shape
    original_metadata['samples'] = W
    original_metadata['lines'] = H
    original_metadata['bands'] = B
    if 'reflectance scale factor' in original_metadata:
        img *= int(float(original_metadata['reflectance scale factor']))
    envi.save_image(out_path, img, metadata=original_metadata, force=True, interleave=original_metadata['interleave'], dtype=np.uint16)
    


