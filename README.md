# MobiSpectral: Hyperspectral Imaging on Mobile Devices
Hyperspectral imaging systems capture information in multiple wavelength bands across the electromagnetic spectrum. These bands provide substantial details based on the optical properties of the materials present in the captured scene. The high cost of hyperspectral cameras and their strict illumination requirements make the technology out of reach for end-user and small-scale commercial applications. We propose MobiSpectral which turns a low-cost phone into a simple hyperspectral imaging system, without any changes in the hardware. We design deep learning models that take regular RGB images and near-infrared (NIR) signals (which are used for face identification on recent phones) and reconstruct multiple hyperspectral bands in the visible and NIR ranges of the spectrum. Our experimental results show that MobiSpectral produces accurate bands that are comparable to ones captured by actual hyperspectral cameras. The availability of hyperspectral bands that reveal hidden information enables the development of novel mobile applications that are not currently possible. To demonstrate the potential of MobiSpectral, we use it to identify organic solid foods, which is a challenging food fraud problem that is currently partially solved by laborious, unscalable, and expensive processes. We collect large datasets in real environments under diverse illumination conditions to evaluate MobiSpectral. Our results show that MobiSpectral can identify organic foods, e.g., apples, tomatoes, kiwis, strawberries, and blueberries, with an accuracy of up to 94% from images taken by phones.

There are three main components of MobiSpectral
- Hyperspectral Reconstruction
- Spectral Classification
- Mobile Application

## Phase 1 : Hyperspectral Reconstruction
- In this phase, we are training & testing hyperspectral reconstruction model using images captured from Hyperspectral camera.
- The input to the deep learning model is RGB & NIR images (4 channels), output is Hyperspectral cubes with ```N``` bands.
  
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
- The dataset is categorized into different fruits, download [[kiwi](https://drive.google.com/file/d/16B9Jnwgo9Xev4db3ROqvL8_64vAr3l-H/view?usp=sharing)] and move it to root folder.
- Each fruit dataset is named ``dataset_{fruit}``, e.g. ``dataset_kiwi``
- Directory structure (e.g. fruit = kiwi)
  ```bash
   |--mobicom23_mobispectral
    |--reconstruction
    |--classification
    |--application 
    |--dataset_kiwi
          |--reconstruction (Ground Truth Hyperspectral data, paired to RGB+NIR)
          |--mobile_data (Paired RGB+NIR mobile images, two classes organic/non-organic)
          |--classification (Reconstructed Hyperspectral from mobile images) 
  ```
### Evaluation on Test Set
- Download the pretrained model [here](https://drive.google.com/file/d/17RGFLNClfeqXwU-uVHdVnYEivxbQ6HrT/view?usp=sharing).
- Move the downloaded folder to the path ```mobicom23_mobispectral/reconstruction/pretrained_models/```
```bash
cd reconstruction/test
# test on kiwi dataset 
python3 test.py --data_root ../../dataset_kiwi/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_kiwi/  --gpu_id 0
```
- Here, the pretrained model produce the inference on RGB+NIR test dataset and compute performance metrics comparing to the ground truth Hyperspectral data.
- Inferenced images (```.mat``` format) are saved at path ```./exp/hs_inference_kiwi/```.
- Performance metrics are printed MRAE, RMSE, SAM, SID, SSIM, PSNR (Reported in Table 1).
- Similarly, repeat the process for other fruits (e.g. [[blueberries](https://drive.google.com/file/d/1jYHs0Q9rnsx58IaHoR0wSvS4Ep0l7IUO/view?usp=sharing)], [[apple]()]).
```bash
# test on apple dataset 
python3 test.py --data_root ../../dataset_apple/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_apple/  --gpu_id 0
# test on blueberries dataset 
python3 test.py --data_root ../../dataset_blueberries/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_blueberries/  --gpu_id 0
```
### Transfer Learning 
Here, we show the evaluation of [[tomato](https://drive.google.com/file/d/1WbQpNG6GFtvjijb9g27n8QE_yDip8tGH/view?usp=sharing)] with and without transfer learning (Reported in Table 2)
```bash
# test on tomato dataset without transfer learning
python3 test.py --data_root ../../dataset_tomato/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_tomato/  --gpu_id 0
# test on tomato dataset with transfer learning
python3 test.py --data_root ../../dataset_tomato/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_tomato_transfer_68ch.pth --outf ./exp/hs_inference_tomato/  --gpu_id 0
```
- Repeat the process for [[strawberries](https://drive.google.com/file/d/1taaiWVIwjy8PtiuxdxNvr2CTWkuhv_Q4/view?usp=sharing)]
### Training
- For training the model from scratch.
- We train our model on three fruits (apple, kiwi, blueberries).
```bash
cd reconstruction/train
python3 train.py --method mst_plus_plus --batch_size 20 --end_epoch 100 --init_lr 4e-4 --outf ./exp/mst_apple_kiwi_blue/ --data_root1 ../../dataset_apple/reconstruction/ --data_root2 ../../dataset_kiwi/reconstruction/ --data_root3 ../../dataset_blueberries/reconstruction/ --patch_size 64 --stride 64 --gpu_id 0
```

## Phase 2 : Spectral Classification
- In this phase, we use the trained model in Phase 1 to reconstruct Hyperspectral from RGB & NIR images captured by mobile (Google Pixel 4).

### Inference on Mobile data
- The organic/non-organic mobile data is at path ```dataset_kiwi/mobile_data/```.
```bash
cd reconstruction/evaluate_mobile
# reconstruct organic kiwi
python3 test.py --data_root ../../dataset_kiwi/mobile_data/organic/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ../../dataset_kiwi/classification/working_organic/  --gpu_id 0
# reconstruct non-organic kiwi
python3 test.py --data_root ../../dataset_kiwi/mobile_data/nonorganic/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ../../dataset_kiwi/classification/working_nonorganic/  --gpu_id 0
```
- The reconstructed data is stored at path  ```dataset_kiwi/classification/```.

### Classification
- Here, we will classify the organic vs non-organic fruit using mobile data (Hyperspectral reconstructed from RGB + NIR)
- Download the pretrained classifiers [here](https://drive.google.com/file/d/1MapCPrTQaRPANhF5x5Jsxs0pU9gb9YFh/view?usp=sharing).
- Move the downloaded folder to the path ```mobicom23_mobispectral/classification/pretrained_classifiers/```
```bash
cd classification
# inference on pretrained model kiwi
python3 evaluate.py --data_root ../dataset_kiwi/classification/ --fruit kiwi --pretrained_classifier ./pretrained_classifiers/MLP_kiwi.pkl

# classify organic vs non-organic kiwi
python3 classify.py --data_root ../dataset_kiwi/classification/ --fruit kiwi
```

- Similarly, repeat the process for other fruits (e.g., apple)
```bash
cd reconstruction/evaluate_mobile
# reconstruct organic apple
python3 test.py --data_root ../../dataset_apple/mobile_data/organic/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ../../dataset_apple/classification/working_organic/  --gpu_id 0
# reconstruct non-organic apple
python3 test.py --data_root ../../dataset_apple/mobile_data/nonorganic/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ../../dataset_apple/classification/working_nonorganic/  --gpu_id 0
```
```bash
cd classification
# inference
python3 evaluate.py --data_root ../dataset_apple/classification/ --fruit apple --pretrained_classifier ./pretrained_classifiers/MLP_apple.pkl
# classify organic vs non-organic apple
python3 classify.py --data_root ../dataset_apple/classification/ --fruit apple
```
## Phase 3 : Mobile Application
