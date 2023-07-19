# MobiSpectral: Hyperspectral Imaging on Mobile Devices
Hyperspectral imaging systems capture information in multiple wavelength bands across the electromagnetic spectrum. These bands provide substantial details based on the optical properties of the materials present in the captured scene. The high cost of hyperspectral cameras and their strict illumination requirements make the technology out of reach for end-user and small-scale commercial applications. We propose MobiSpectral which turns a low-cost phone into a simple hyperspectral imaging system, without any changes in the hardware. We design deep learning models that take regular RGB images and near-infrared (NIR) signals (which are used for face identification on recent phones) and reconstruct multiple hyperspectral bands in the visible and NIR ranges of the spectrum. Our experimental results show that MobiSpectral produces accurate bands that are comparable to ones captured by actual hyperspectral cameras. The availability of hyperspectral bands that reveal hidden information enables the development of novel mobile applications that are not currently possible. To demonstrate the potential of MobiSpectral, we use it to identify organic solid foods, which is a challenging food fraud problem that is currently partially solved by laborious, unscalable, and expensive processes. We collect large datasets in real environments under diverse illumination conditions to evaluate MobiSpectral. Our results show that MobiSpectral can identify organic foods, e.g., apples, tomatoes, kiwis, strawberries, and blueberries, with an accuracy of up to 94% from images taken by phones.

There are three main components of MobiSpectral
- Hyperspectral Reconstruction
- Spectral Classification
- Mobile Application

## Phase 1 : Hyperspectral Reconstruction
### Prerequisites
- Linux or macOS
- Python 3 (Anaconda)
- NVIDIA GPU + CUDA CuDNN

### Installation
- Clone this repo & install dependencies:
```bash
git clone https://github.com/mobispectral/mobicom23_mobispectral.git
cd mobicom23_mobispectral
pip install -r requirements.txt
```
### Dataset
### Evaluation on Validation Set
### Inference on Mobile data
### Training

## Phase 2 : Spectral Classification
## Phase 3 : Mobile Application
