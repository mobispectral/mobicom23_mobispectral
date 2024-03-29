# MobiSpectral: Hyperspectral Imaging on Mobile Devices
This repository describes the detailed steps to reproduce the research results presented in the paper titled: 
[``MobiSpectral: Hyperspectral Imaging on Mobile Devices``](https://dl.acm.org/doi/10.1145/3570361.3613296) published in ACM MobiCom'23. 

There are three main components of MobiSpectral to evaluate: 
- Hyperspectral Reconstruction
- Identification of Organic Fruits 
- [Mobile Application](https://github.com/mobispectral/MobiSpectral-Android)

## Hyperspectral Reconstruction
- MobiSpectral has a hyperspectral reconstruction model that was trained on images captured by a hyperspectral camera.
- You can reproduce our results using the pre-trained model. Alternatively, you can start by training the model from scratch, but this may take several hours.
  
### Prerequisites
- Workstation running Linux
- NVIDIA GPU + CUDA CuDNN
- Python Anaconda (Python version 3.8 or earlier) 

### Install the code 
- Clone the following repo, create Anaconda environment, and install [Pytorch](https://pytorch.org/get-started/previous-versions/) & other dependencies:
```bash
git clone https://github.com/mobispectral/mobicom23_mobispectral.git
cd mobicom23_mobispectral
pip install -r requirements.txt
```

The above instructions should work for most environments. However, for some CUDA drivers, you may need to do the following more specific instructions instead: 

1. Download miniconda from https://docs.conda.io/en/latest/miniconda.html  (Choose Linux Python 3.8)
2. Install miniconda:
```bash
bash Miniconda3-py38_23.5.2-0-Linux-x86_64.sh
```

3. Clone repo & create anaconda env
  ```bash
  git clone https://github.com/mobispectral/mobicom23_mobispectral.git
  cd mobicom23_mobispectral
  conda create --name MobiSpec python=3.8
  conda activate MobiSpec
```

4. Install pytorch & dependencies
  ```bash
  pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirements.txt
 ```


After installation, the directory structure should look like:

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
- There are five datasets for different fruits: apple, kiwi, tomato, blueberries, and strawberries. Each is named as ``dataset_{fruit}``, e.g., ``dataset_kiwi``. 
- The directory structure of the datasets looks like: 
  ```bash
   |--datasets
      |--dataset_kiwi
          |--reconstruction (Ground Truth Hyperspectral data, paired to RGB+NIR)
          |--mobile_data (Paired RGB+NIR mobile images, two classes organic/non-organic)
          |--classification (Reconstructed hyperspectral images from mobile images)
       |--dataset_apple
          |--...
      ... 
  ```
- To evaluate the reconstruction mode, you will need to download one or more of the following five datasets:

     - [kiwi](https://drive.google.com/file/d/1PHsMs3TtQYg-VmhrJKfy-jzplUgHNDWx/view) (2.6 GB)
  
    - [blueberries](https://drive.google.com/file/d/1PF-yzTW3ao6ZACLPlOeO_Z-qiUkBcxz3/view) (1.9 GB)

    - [apple](https://drive.google.com/file/d/1PFiOQtyRwSCSV6gIpfZ9beoyWQDgsZN1/view) (19.0 GB)
    
    - [tomato](https://drive.google.com/file/d/1PELLiBpeNmgrQHDeuWdWWvCMfBt4dk9r/view) (1.7 GB)
  
    - [strawberries](https://drive.google.com/file/d/1PI5505giSb4LLBTVCMvT318wWONPkJEZ/view) (2.0 GB)

- Unzip the downloaded dataset(s) and move it (them) to the datasets folder. Please note that additional storage (similar in size to the downloaded dataset) will be needed to reproduce the reconstruction results.
   
- You can also download [all datasets together](https://doi.org/10.20383/103.0811) (27 GB). 

 
### Reproduce the reconstruction results using the pre-trained model
- Download the pretrained model [here](https://drive.google.com/file/d/1P7LcvEvrV-8Mr-QzyuqN9v5f5jzyb_a-/view) (about 250 MB).
- Move the .zip file to the ```mobicom23_mobispectral/``` folder and unzip it there.
- Test the reconstruction model on the kiwi dataset as follows: 
```bash
cd reconstruction/test
# test on kiwi dataset 
python3 test.py --data_root ../../datasets/dataset_kiwi/reconstruction/  --method mst_plus_plus --pretrained_model_path ../../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_kiwi/  --gpu_id 0
```
- Here, the pretrained model produces the inference on the test partition of the kiwi dataset and computes multiple performance metrics to compare against the ground truth hyperspectral data.
- Inferenced images (```.mat``` format) are saved at path ```./exp/hs_inference_kiwi/```.
- The following performance metrics are printed: MRAE, RMSE, SAM, SID, SSIM, and PSNR (These are the ones reported in Table 1 in the paper). 
- Similarly, you can repeat the process for other fruits (e.g. blueberries, apple).
```bash
# test on apple dataset 
python3 test.py --data_root ../../datasets/dataset_apple/reconstruction/  --method mst_plus_plus --pretrained_model_path ../../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_apple/  --gpu_id 0
# test on blueberries dataset 
python3 test.py --data_root ../../datasets/dataset_blueberries/reconstruction/  --method mst_plus_plus --pretrained_model_path ../../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_blueberries/  --gpu_id 0
```
### Transfer learning 
Here, we show the evaluation of ``dataset_tomato`` with and without transfer learning (Reported in Table 2). 
```bash
# test on tomato dataset without transfer learning
python3 test.py --data_root ../../datasets/dataset_tomato/reconstruction/  --method mst_plus_plus --pretrained_model_path ../../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ./exp/hs_inference_tomato/  --gpu_id 0
# test on tomato dataset with transfer learning
python3 test.py --data_root ../../datasets/dataset_tomato/reconstruction/  --method mst_plus_plus --pretrained_model_path ../../pretrained_models/mst_tomato_transfer_68ch.pth --outf ./exp/hs_inference_tomato/  --gpu_id 0
```
- Repeat the process for ``dataset_strawberries``

### Training the reconstruction model from scratch
- This may take several hours, depending on the GPU.
- To train the model on three fruits (apples, kiwis, and blueberries):
```bash
cd reconstruction/train
python3 train.py --method mst_plus_plus --batch_size 20 --end_epoch 100 --init_lr 4e-4 --outf ./exp/mst_apple_kiwi_blue/ --data_root1 ../../datasets/dataset_apple/reconstruction/ --data_root2 ../../datasets/dataset_kiwi/reconstruction/ --data_root3 ../../datasets/dataset_blueberries/reconstruction/ --patch_size 64 --stride 64 --gpu_id 0
```

## Identification of Organic Fruits
- We use the trained reconstruction model to reconstruct hyperspectral bands from RGB & NIR images captured by a mobile phone (Google Pixel 4 XL).
- Then, we feed the reconstructed hyperspectral bands to a classifier to distinguish organic fruits from non-organic ones. 

### Hyperspectral reconstruction of the mobile image dataset 
- The organic/non-organic mobile data is at ```dataset_kiwi/mobile_data/```.
```bash
cd reconstruction/evaluate_mobile
# reconstruct organic kiwi
python3 test.py --data_root ../../datasets/dataset_kiwi/mobile_data/organic/  --method mst_plus_plus --pretrained_model_path ../../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ../../datasets/dataset_kiwi/classification/working_organic/  --gpu_id 0
# reconstruct non-organic kiwi
python3 test.py --data_root ../../datasets/dataset_kiwi/mobile_data/nonorganic/  --method mst_plus_plus --pretrained_model_path ../../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ../../datasets/dataset_kiwi/classification/working_nonorganic/  --gpu_id 0
```
- The reconstructed data is stored at ```dataset_kiwi/classification/```.

### Organic classification
- We classify the organic and non-organic fruits using the reconstructed bands from the RGB + NIR images captured by the phone.

```bash 
cd classification
# inference on pretrained model kiwi
python3 evaluate.py --data_root ../datasets/dataset_kiwi/classification/ --fruit kiwi --pretrained_classifier ../pretrained_models/MLP_kiwi.pkl

# classify organic vs non-organic kiwi
python3 classify.py --data_root ../datasets/dataset_kiwi/classification/ --fruit kiwi
```

- Similarly, repeat the process for other fruits (e.g., apple)
```bash
cd reconstruction/evaluate_mobile
# reconstruct organic apple
python3 test.py --data_root ../../datasets/dataset_apple/mobile_data/organic/  --method mst_plus_plus --pretrained_model_path ../../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ../../datasets/dataset_apple/classification/working_organic/  --gpu_id 0
# reconstruct non-organic apple
python3 test.py --data_root ../../datasets/dataset_apple/mobile_data/nonorganic/  --method mst_plus_plus --pretrained_model_path ../../pretrained_models/mst_apple_kiwi_blue_68ch.pth --outf ../../datasets/dataset_apple/classification/working_nonorganic/  --gpu_id 0
```
```bash
cd classification
# inference
python3 evaluate.py --data_root ../datasets/dataset_apple/classification/ --fruit apple --pretrained_classifier ../pretrained_models/MLP_apple.pkl
# classify organic vs non-organic apple
python3 classify.py --data_root ../datasets/dataset_apple/classification/ --fruit apple
```

## Mobile Application [[link](https://github.com/mobispectral/MobiSpectral-Android)]

## Citation
If you use our code or dataset for your research, please cite our paper.
```
@inproceedings{10.1145/3570361.3613296,
author = {Sharma, Neha and Waseem, Muhammad Shahzaib and Mirzaei, Shahrzad and Hefeeda, Mohamed},
title = {MobiSpectral: Hyperspectral Imaging on Mobile Devices},
year = {2023},
isbn = {9781450399906},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3570361.3613296},
doi = {10.1145/3570361.3613296},
booktitle = {Proceedings of the 29th Annual International Conference on Mobile Computing and Networking},
articleno = {82},
numpages = {15},
keywords = {mobile applications, food fraud, hyperspectral imaging},
location = {Madrid, Spain},
series = {ACM MobiCom '23}
}
```
