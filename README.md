# MobiSpectral: Hyperspectral Imaging on Mobile Devices

This repository describes the detailed steps to reproduce the research results presented in the paper titled: 
``MobiSpectral: Hyperspectral Imaging on Mobile Devices``.  

There are three main components of MobiSpectral to evaluate: 
- Hyperspectral Reconstruction
- Classification of Organic Fruits 
- Mobile Application

## Hyperspectral Reconstruction
- MobiSpectral has a hyperspectral reconstruction model that was trained on images captured by a hyperspectral camera.
- You can [reproduce our results using the pre-trained model](### Reproduce the Reconstruction Results using the Pre-trained Model). Alternatively, you can start by training the model from scratch, but this may take several hours and require downloading multiple extra gigabytes of training data.
  
### Prerequisites
- Workstation running Linux or MacOS
- NVIDIA GPU + CUDA CuDNN
- Python <= 3.8 (Anaconda)

### Instal the code 
- Clone this repo & install dependencies:
```bash
git clone https://github.com/mobispectral/mobicom23_mobispectral.git
cd mobicom23_mobispectral
pip install -r requirements.txt
```

XXX Mohamed: structure the directory as follows: 

```bash
   |--mobicom23_mobispectral
      |--reconstruction
      |--classification
      |--application 
      |--datasets
      |--pretrained_models
  
```
The datasets and pretrained_models folders are initially empty. 

### Download datasets
- The datasets are categorized into different fruits; each is named as ``dataset_{fruit}``, e.g., ``dataset_kiwi``. 
- The directory structure looks like the following: 
  ```bash
   |--datasets
      |--dataset_kiwi
          |--reconstruction (Ground Truth Hyperspectral data, paired to RGB+NIR)
          |--mobile_data (Paired RGB+NIR mobile images, two classes organic/non-organic)
          |--classification (Reconstructed Hyperspectral from mobile images)
       |--dataset_apple
          |--...
  ```

- XXX Download [[kiwi](https://drive.google.com/file/d/16B9Jnwgo9Xev4db3ROqvL8_64vAr3l-H/view?usp=sharing)] and move it to root folder.

### Reproduce the Reconstruction Results using the Pre-trained Model
- Download the pretrained model [here](https://drive.google.com/file/d/17RGFLNClfeqXwU-uVHdVnYEivxbQ6HrT/view?usp=sharing) (about 250 MB).
- Move the downloaded folder to the path ```mobicom23_mobispectral/reconstruction/pretrained_models/```
```bash
cd reconstruction/test
# test on kiwi dataset 
python3 test.py --data_root ../../dataset_kiwi/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_kiwi/  --gpu_id 0
```
- Here, the pretrained model produces the inference on the RGB+NIR test dataset and computes performance metrics to compare against the ground truth hyperspectral data.
- Inferenced images (```.mat``` format) are saved at path ```./exp/hs_inference_kiwi/```.
- The following performance metrics are printed: MRAE, RMSE, SAM, SID, SSIM, and PSNR (These are the ones reported in Table 1 in the paper). 
- Similarly, repeat the process for other fruits (e.g. [[blueberries](https://drive.google.com/file/d/1jYHs0Q9rnsx58IaHoR0wSvS4Ep0l7IUO/view?usp=sharing)], [[apple](https://drive.google.com/file/d/1WtogFi1ahG5ejzpcp0GcUs64MEuQDJjT/view?usp=sharing)]).
```bash
# test on apple dataset 
python3 test.py --data_root ../../dataset_apple/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_apple/  --gpu_id 0
# test on blueberries dataset 
python3 test.py --data_root ../../dataset_blueberries/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_blueberries/  --gpu_id 0
```
### Transfer Learning 
Here, we show the evaluation of [[tomato](https://drive.google.com/file/d/1WbQpNG6GFtvjijb9g27n8QE_yDip8tGH/view?usp=sharing)] with and without transfer learning (Reported in Table 2). 
```bash
# test on tomato dataset without transfer learning
python3 test.py --data_root ../../dataset_tomato/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_tomato/  --gpu_id 0
# test on tomato dataset with transfer learning
python3 test.py --data_root ../../dataset_tomato/reconstruction/  --method mst_plus_plus --pretrained_model_path ../pretrained_models/mst_tomato_transfer_68ch.pth --outf ./exp/hs_inference_tomato/  --gpu_id 0
```
- Repeat the process for [[strawberries](https://drive.google.com/file/d/1taaiWVIwjy8PtiuxdxNvr2CTWkuhv_Q4/view?usp=sharing)]

### Training the model from scratch
- This may take several hours, depending on the GPU.
- We train our model on three fruits (apples, kiwis, and blueberries).
```bash
cd reconstruction/train
python3 train.py --method mst_plus_plus --batch_size 20 --end_epoch 100 --init_lr 4e-4 --outf ./exp/mst_apple_kiwi_blue/ --data_root1 ../../dataset_apple/reconstruction/ --data_root2 ../../dataset_kiwi/reconstruction/ --data_root3 ../../dataset_blueberries/reconstruction/ --patch_size 64 --stride 64 --gpu_id 0
```

## Spectral Classification
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
