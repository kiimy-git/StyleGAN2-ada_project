# Creating Various Images using StyleGAN2-ada(Hispanic)
### **Team Project**
### 목표
* GAN부터 styleGAN2-ada까지 모델의 목적과 방향성을 학습
* StyleGAN2-ada을 활용하여 세상에 존재할법한 이미지를 생성

[nvidia styleGAN2-ada](https://github.com/NVlabs/stylegan2-ada)

### 문제인식
* 수 많은 데이터가 존재하지만 정말 유의미한 데이터 확보 어려움

### Benefit
* GAN은 real data의 학습분포를 학습하며 유의미한 data를 생성해낸다는 부분에서 매우 매력적인 기술
* **GAN을 활용하여 유의미한 데이터를 확보하는데 수월해지고 이 데이터들을 가지고 AI모델 성능 향상에 도움**
* **데이터를 확보하는 데 소비되는 비용과 시간을 절약**

# Requirements
[clone styleGAN2-ada](https://github.com/dvschultz/stylegan2-ada)

**SOTA모델을 사용해보는 경험도 중요하지만 먼저 GAN이 어떤 과정으로 학습이 되는지 먼저 알아야할 필요가 있다**

[PPT - GAN은 무엇인가? styleGAN2-ada란?](https://github.com/kimmy-git/StyleGAN2-ada_project/blob/main/AI_04_%EA%B9%80%EC%98%81%ED%9B%88_styleGAN2-ada(ppt).pptx)


# Tools
* numpy
* OpenCV
* Tensorflow 1.x
* os
* PIL
* tqdm
* pathlib
* glob
* matplotlib
* opensimplex

# Process
- [Getting started](#getting-started)
- [Data](#data)
- [Metrics and score](#metrics-and-score)
- [Train Process](#train-process)
- [Image Generate](#image-generate)
  * [Grid](#grid)
  * [Average image](#average-image)
  * [Interpolation](#interpolation)
  * [style Mixing](#style-mixing)
- [Results](#results)
- [Reviews](#reviews)

# Getting started
```python
#!git clone https://github.com/dvschultz/stylegan2-ada (= StyleGAN2-ada)

import os
if os.path.isdir("/content/drive/MyDrive/AIB_04_Stylebot/colab-sg2-ada"):
    %cd "/content/drive/MyDrive/AIB_04_Stylebot/colab-sg2-ada/stylegan2-ada"
else:
    #install script
    %cd "/content/drive/MyDrive/AIB_04_Stylebot"
    !mkdir colab-sg2-ada
    %cd colab-sg2-ada
    !git clone https://github.com/dvschultz/stylegan2-ada
    %cd stylegan2-ada
    !mkdir downloads
    !mkdir datasets
```

# Data
* Pinterest, Google, SNS..등 Hispanic 여성의 이미지 수집(유명인사 위주로)
* Hispanic 여성마다 특징을 분류
* 2번째 여성이미지(55장 학습 진행)
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144360626-04d71ea4-6110-465e-898c-82bb6e85b8d2.png"></p>

# Metrics and score
- FID(Frechet Inception Distance)
- PPL(Pereceptual Path Length)

**StyleGAN에선 추가적으로 PPL을 제안하지만 프로젝트의 주 목적은 score보다 데이터셋을 통해서 유의미한 이미지 생성 후 육안으로 확인**

# Train Process 
[.ipynb 참고](https://github.com/kimmy-git/StyleGAN2-ada_project/blob/main/AI_04_%EA%B9%80%EC%98%81%ED%9B%88_styleGAN2-ada.ipynb)
### 1. image resize(1024 * 1024)진행
- 얼굴부분만 crop 진행 후 고해상도 이미지를 생성하기위한 resize(1024*1024)

### 2. Convert dataset to .tfrecords
- TFRecord는 데이터 세트의 포맷의 하나로 TFRecord형식은 바이너리 코드의 시리즈를 저장하기 위한 단순한 형식

### 3. Train a custom model(= Discriminator augmentation 진행) => **"bg"**

[styleGAN2-ada 논문](https://arxiv.org/abs/2006.06676)
```python
augpipe_specs = {
        'blit':     dict(xflip=1, rotate90=1, xint=1),
        'geom':     dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':    dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter':   dict(imgfilter=1),
        'noise':    dict(noise=1),
        'cutout':   dict(cutout=1),
        'bg':       dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':      dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

```
### 4. Transfer learning
- 이미 1024의 이미지가 학습된 ffhq1024 초기 학습 진행
- 이후 학습한 모델.pkl **추가적으로 학습** (colab GPU 한계)
### 5. image Generate(**= install opensimplex**)
```python
!pip install opensimplex
'''Seed must be between 0 and 2**32 - 1'''
network = '/content/drive/MyDrive/AIB_04_Stylebot/network-snapshot-000200.pkl'
!python generate.py generate-images --outdir='/content/image' --trunc=0 --seeds=0-555 --network={network} 

```
### 6. Style-mixing을 사용하여 특정 image 특징을 가져와 합성
```python
!python style_mixing.py --outdir='/content/drive/MyDrive/AIB_04_Stylebot/colab-sg2-ada/style_mixing/style=0-3 1024' --trunc=0.5 --rows=4585 --cols=3333,1548,5184,6954,4878  --network={network} --styles=0-3" 
```

# Image Generate
### (1024*1024) Image
## Grid
* Truncation = 0.5
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144365774-a8d851c4-74d3-4ff4-af68-1b0dd3974a95.gif"></p>

## Average image
* Truncation = 0
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144366291-a7a61bd6-5ff7-48fd-8a0c-ef4eaf032b8e.png" width="50%" height="50%"/></p>

## Interpolation
https://user-images.githubusercontent.com/83389640/177293268-c2fe90d7-08eb-4056-9a16-c29105461c85.mp4

## style Mixing
* column => 0-3 layer의 특징 + low
<p align="center"><img src="https://user-images.githubusercontent.com/83389640/144366538-9c5e36bb-cb5f-42e6-aaef-4583ec0952bf.png"></p>

# Results
1. StyleGAN2-ada는 적은 데이터로도 Augmentation을 통해서 모델 학습
2. Truncation을 사용하여 유의미한 데이터만 생성
3. Style-mixing 기능을 사용하여 원하는 data에 특징을 부여하여 새로운 data 생성
4. 데이터에 대한 저작권을 피할 수 있을 것
5. 데이터를 확보하는 데 소비되는 비용과 시간을 절약

# Reviews
1. 학습이 너무 오래걸린다. -> 이미 충분히 학습을 했던 모델이기에 성능은 나쁘지 않았다.(그만큼 시간소요)
2. Generate Image가 데이터셋 image와의 유사성이 보임. -> 너무 적은데이터로 Augmentation을 진행했기 때문에?
3. Generate 된 Image는 과연 저작권이 없을 수 있다 볼 수 있는가? -> StyleGAN2에선 Inversion을 제안(이해X)
4. styleGAN 구조에서 8개의 FC는 어떻게 데이터의 상관관계를 줄여주는가?

**결과적으로 GAN의 목적과 방향성을 공부한 것은 도움이 되었지만 구현 방법에 대해 정확히 다 이해하기에는 한계**

**간단한 구조부터 구현해보며 이해하기**
[밑바닥 부터 시작하는 딥러닝]()
