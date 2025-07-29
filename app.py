import streamlit as st 
import pathlib
import numpy as np
import click
import rasterio
import pywt
from scipy import interpolate
import scipy.ndimage
from tqdm import tqdm
from PIL import Image
import cv2
import random as r
page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://p.turbosquid.com/ts-thumb/5r/uFLMl6/Xv/moonturntable/jpg/1610787087/1920x1080/turn_fit_q99/812067da572188f09f548c426e61bc59b8613abd/moonturntable-1.jpg");
  background-size: cover;
}
[data-testid="stHeader"]{
  background-color: rgba(0,0,0,0);
  }
  [data-testid="stScreencast"]{
  color: rgba(0,0,0,0);
  }
  
</style>
"""
infile = pathlib.Path(r"D:\SIH(again)\image_cleaning\op.tif")
outfile = pathlib.Path(r"D:\SIH(again)\image_cleaning\output\op.tif")
st.markdown(page_element, unsafe_allow_html=True)
st.title("SPACE PENGUINS")
st.header("DENOISING")


WAVELETS = [wave for fam in pywt.families() for wave in pywt.wavelist(family=fam)]


def damp_coefficient(coeff, sigma):
    fft_coeff = np.fft.fft(coeff, axis=0)
    fft_coeff = np.fft.fftshift(fft_coeff, axes=[0])

    ydim, _ = fft_coeff.shape
    gauss1d = 1 - np.exp(-np.arange(-ydim // 2, ydim // 2)**2 / (2 * sigma**2))
    damped_fc = fft_coeff * gauss1d[:, np.newaxis]

    damped_coeff = np.fft.ifftshift(damped_fc, axes=[0])
    damped_coeff = np.fft.ifft(damped_coeff, axis=0)
    return damped_coeff.real

def gamma_correction(img, gamma=1.2):
    
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    
    if len(img.shape) < 3 or img.shape[2] != 3:
        if len(img.shape) == 3 and img.shape[2] == 4:  
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:  
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

   
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    
    l, a, b = cv2.split(lab)

   
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

   
    lab_clahe = cv2.merge((l_clahe, a, b))

 
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return img_clahe

def sharpen_image(img, sigma=1.0, alpha=1.5, beta=-0.5):
    
    blurred = cv2.GaussianBlur(img, (0,0), sigma)
    sharp = cv2.addWeighted(img, alpha, blurred, beta, 0)
    return sharp


def remove_stripes(image, decomp_level, wavelet, sigma):
    coeffs = pywt.wavedec2(
        image, wavelet=wavelet, level=decomp_level, mode='symmetric')

    damped_coeffs = [coeffs[0]]

    for ii in range(1, len(coeffs)):
        ch, cv, cd = coeffs[ii]

        cv = damp_coefficient(cv, sigma)
        ch = damp_coefficient(ch, sigma)

        damped_coeffs.append((ch, cv, cd))

    rec_image = pywt.waverec2(damped_coeffs, wavelet=wavelet, mode='symmetric')
    return rec_image


def rmstripes(wavelet='db10', decomp_level=8, sigma=8, band=1, show_plots=True):
    """Remove stripes from an image by applying a wavelet-FFT approach."""
    with np.errstate(invalid='ignore'):
        with rasterio.open(infile.as_posix(), 'r') as src:
            profile = src.profile.copy()
            image = src.read(band)

    click.echo('Removing stripes ...')
    no_stripes = remove_stripes(image, decomp_level, wavelet, sigma)


    if no_stripes.dtype == np.float64:
        no_stripes = np.clip(no_stripes, 0, 255).astype(np.uint8)

    profile.update({
        'dtype': no_stripes.dtype,
        'count': 1
    })
    with rasterio.open(outfile.as_posix(), 'w', **profile) as dst:
        dst.write(no_stripes, 1)

    if show_plots:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(image, cmap='gray')
        ax[1].imshow(no_stripes, cmap='gray')
        plt.show()
    img = cv2.imread(infile)
    img1 = cv2.imread(outfile)
    brightened = gamma_correction(img1, gamma=1.2)
    clahe_img = apply_clahe(brightened, clip_limit=2.0, tile_grid_size=(8,8))
    sharpened = sharpen_image(clahe_img, sigma=1.0, alpha=1.5, beta=-0.5)
    cv2.imwrite("D:\SIH(again)\image_cleaning\output\opp.tif",sharpened)
    cols = st.columns(4)
    images = [uploaded_image,img,img1,sharpened]
    for i, img in enumerate(images):
        cols[i % 4].image(img, width=1200)

uploaded_image = st.file_uploader("Choose a png,  jpg  or  jpeg file",type=["png","jpg","jpeg","tif"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    uploaded_image = np.array(image)
    im = cv2.cvtColor(uploaded_image, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(im)
    clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(10,10))
    enhanced_b = clahe.apply(b)
    enhanced_g = clahe.apply(g)
    enhanced_r = clahe.apply(r)
    enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
    gamma_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    enhanced_img = np.clip((enhanced_img / 255.0) ** gamma_values[8] * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(r"D:\SIH(again)\image_cleaning\op.tif",enhanced_img)
    rmstripes()

import io
img_byte_arr = io.BytesIO()
img1 = cv2.imread(outfile)
enhanced_img = Image.fromarray(img1)
enhanced_img.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

btn = st.download_button(
    label="Download Image",
    data=img_byte_arr,  
    file_name="image.tif",
    mime="image/tif"
)

