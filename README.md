# 🤸‍♂️Image Classification

✔생생한 Daily Contributions는 [이곳](https://www.notion.so/iloveslowfood/Stage-2-Image-Classification-58dbfca2e1ef4e36b8de6790b403ccba)에 업로드되어 있습니다:)



## Task Description

- ***Period.*** 2021.03.29~2021.04.08
- ***Problem Type.*** Classification - 마스크/성별/연령대에 따른 18개 클래스
- ***Metric.*** Macro F1 Score
- ***Data.*** 한 명당 7장(마스크 착용x1, 미착용x1, 불완전 착용x5) ,총 *2*,700명의 이미지. 한 사람당 384x512



## Performances

### Scores

- Public LB.  F1 0.7706, Accuracy 81.3333%
- Private LB. F1 0.7604, Accuracy 81.0952%

### Best Model Configuration

#### I. Structure: *Ensemble VanillaEfficientNet with K-Fold CV*

K-Fold CV를 통해 학습된 VanillaEfficientNet 모델을 각 Fold별로 N개씩 저장, 추론 단계에서 각 Fold별 K개의 모델을 불러와 TTA(Test Time Augmentation)를 적용하여 모든 결과값을 앙상블했습니다. Fold 수를 5로, Fold별 불러올 모델 수를 2로, TTA를 2로 설정하여 총 20(5x2x2)개의 추론 결과를 산술평균한 Soft Voting 앙상블의 성능이 가장 높았습니다. (Private LB. F1 0.7604, Accuracy 81.0952%)

![ensemble_1](https://github.com/iloveslowfood/ImageClassfication/blob/main/etc/ensemble_1.png?raw=true)

![ensemble_2](https://github.com/iloveslowfood/ImageClassfication/blob/main/etc/ensemble_2.png?raw=true)

#### II. Hyper Parameters

앙상블에 활용한 VanillaEfficientNet 모델의 하이퍼 파라미터는 다음과 같습니다.

```python
batch_size=32
epochs=모델 별 상이
loss_type='labelsmoothingLoss'
lr=0.001
lr_scheduler='cosine' # cosine annealing warm restart
model_type='VanillaEfficientNet'
optim_type='adam'
seed=42
transform_type='tta'

# 'tta' transform
# train phase
transforms.Compose(
    [
        transforms.CenterCrop((384, 384)),
        transforms.RandomResizedCrop((224, 224)),
        RandAugment(2, 9), # (N, M): (# of transform candidates, # of changes)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# test phase
transforms.Compose(
    [
        transforms.CenterCrop((384, 384)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
```



## Command Line Interface

### Train Phase

```python
>>> python train.py --task 'main' --model-type 'VanillaEfficientNet' --cv 5
```

18가지 카테고리를 분류하는 Main Task 이외에 마스크 상태, 연령대(classification) 및 연령(regression), 성별의 4가지 sub task에 대한 학습을 모두 지원하며, K-Fold CV 학습 또한 가능합니다. 조정 가능한 argument는 다음과 같습니다.

- `task` : main task(*main*), 마스크 상태(*mask*), 연령대(*ageg*), 연령(*age*), 성별(*gender*)의 5가지 task에 대한 학습이 가능합니다. (default: *main*)

- `model_type` : 불러올 모델 아키텍쳐를 선택합니다. 지원하는 모델 아키텍쳐는 ***VanillaEfficientNet***(`'VanillaEfficientNet'`), ***VanillaResNet***(`'VanillaResNet'`), ***MultiLabelTHANet*** (`'MultiLabelTHANet'`), ***MultiClassTHANet***(`'MultiClassTHANet_MK1'`), ***THANet***(`'THANet_MK1'`)이 있습니다. (default: `'VanillaEfficientNet'`)

- `load_state_dict`: 저장된 모델을 불러와 학습할 경우 저장된 파일의 경로를 입력합니다. 저장된 파라미터와 `model_type`이 일치해야 합니다.

- `train/valid_root`: 학습용 데이터와 검증용 데이터의 경로를 입력합니다.

- `transform_type`: Augmentation에 활용할 Transform 종류를 입력합니다. ***Base***(`'base'`), ***Random***(`'random'`), ***TTA***(`'tta'`), ***Face Crop***(`'facecrop'`)을 지원합니다.

- `age_filter`: 일정 나이 이상인 인물의 연령대를 60대 이상 연령대로 강제 변경합니다. 50대 후반의 인물과 60대 인물의 사진은 분간하기가 어려워 예측 성능에 지장을 주는 경우가 있었기 때문에 고안한 argument입니다. 가령,  `age_filter=58`로 설정 시 58세 이상인 인물 모두가 '60대 이상 연령대' 범주로 강제 변경됩니다.

- `epochs`: 에폭을 설정합니다. (default: 30)

- `cv`: K-Fold CV를 활용할 경우 사용하는 argument로, Fold 수를 설정합니다. 입력하지 않거나 1을 입력할 경우 단일 폴드로 학습, 즉 K-Fold CV가 진행되지 않습니다. (default: 1)

- `batch_size`: 배치 사이즈를 설정합니다. (default: 32)

- `optim_type`: 최적화 함수를 설정합니다. ***Adam***, ***AdamP***, ***SGD***, ***Momentum***을 지원합니다. (default: `adam`)

- `loss_type`: 손실 함수를 설정합니다. ***Label Smoothing Loss***(`'labelsmoothingloss'`), ***Focal Loss***(`'focalloss'`), ***Cross Entropy Loss***(`'crossentropyloss'`), ***MSE Loss***(`'mseloss'`), ***Smooth L1 Loss***(`'smoothl1loss'`)를 지원합니다. (default: `labelsmootingloss`)

- `lr`: Learning Rate를 설정합니다. (default: `0.005`)

- `lr_scheduler`: LR 스케줄러를 설정합니다. ***Cosine Annealing LR Decay***(`'cosine'`)를 지원합니다. (default: `cosine`)

  

### Inference Phase

#### I. Singular Model Inference

```python
>>> python submit_singular --task 'main' --model-type 'VanillaEfficientNet' --transform-type 'random'
```

- `task` : 메인 task(`'main'`), 마스크 상태(`'mask'`), 연령대(`'ageg'`), 연령(`'age'`), 성별(`'gender'`)의 5가지 task에 대한 추론이 가능합니다. (default: `'main'`)

- `model_type` : 불러올 모델 아키텍쳐를 선택합니다. 지원하는 모델 아키텍쳐는 ***VanillaEfficientNet***(`'VanillaEfficientNet'`), ***VanillaResNet***(`'VanillaResNet'`), ***MultiLabelTHANet*** (`'MultiLabelTHANet'`), ***MultiClassTHANet***(`'MultiClassTHANet_MK1'`), ***THANet***(`'THANet_MK1'`)이 있습니다. (default: `'VanillaEfficientNet'`)

- `load_state_dict` : 추론에 활용할 사전 학습된 파라미터 파일의 경로를 설정합니다. 모델 아키텍쳐에 맞는 파라미터 파일을 불러와야 정상 작동합니다.

- `transform_type`: Augmentation에 활용할 Transform 종류를 입력합니다. ***Base***(`'base'`), ***Random***(`'random'`), ***TTA***(`'tta'`), ***Face Crop***(`'facecrop'`)을 지원하며, 각 transform은 다음과 같습니다. (default: ***Base***(`'base'`))

- `data_root`: 추론할 데이터의 경로를 입력합니다. (default: `./input/data/images`)

- `save_path`: 추론 결과를 저장할 경로를 입력합니다. 추론 결과는 ImageID와 ans의 두 컬럼을 포함한 csv 파일 형태로 저장됩니다. (default: `'./prediction'`)

  

#### II. Ensemble Inference

```python
>>> python submit_ensemble --task 'main' --root './saved_ensemble_models' --transform-type --method 'soft' --top-k 3 --tta 2
```

- `task` : 메인 task(`'main'`), 마스크 상태(`'mask'`), 연령대(`'ageg'`), 연령(`'age'`), 성별(`'gender'`)의 5가지 task에 대한 추론이 가능합니다. (default: `'main'`)

- `root` : 앙상블할 모델이 저장된 폴더 경로를 설정합니다. 현재 KFold 기반의 앙상블만이 지원되기 때문에, 입력한 경로는 다음과 같은 디렉토리 구조를 가져야 합니다. K-Fold의 최상위 폴더의 이름을 기준으로 모델 아키텍쳐를 불러오기 때문에, 최상위 폴더에는 모델명이 반드시 기재되어야 합니다. 

  ```shell
  kfold-ensemble-VanillaEfficientNet
   ├─fold00
   ├─fold01
   ├─fold02
   ├─ ...
   └─foldNN
  ```

- `transform_type`: Augmentation에 활용할 Transform 종류를 입력합니다. ***Base***(`'base'`), ***Random***(`'random'`), ***TTA***(`'tta'`), ***Face Crop***(`'facecrop'`)을 지원하며, 각 transform은 다음과 같습니다. (default: ***Base***(`'base'`))

- `data_root`: 추론할 데이터의 경로를 입력합니다. (default: `'./input/data/images'`)

- `top_k`: 각 Fold별로 몇 개의 모델을 앙상블에 활용할 지 설정합니다. Fold별 검증 성능이 가장 높았던 모델부터 불러옵니다. 가령, `top_k = 2`로 설정할 경우, 각 Fold로부터 성능이 가장 좋은 2개의 모델을 불러와 총 `(# of Folds) x 2` 번의 추론을 진행, 앙상블합니다. (default: `3`)

- `method`: 앙상블 방식을 설정합니다. Hard Voting 방식(`'hardvoting'`)과 Soft Voting 방식(`'softvoting'`)이 있으며, Soft Voting 설정 시 산술평균(`arithmetic`), 기하평균(`geometric`), 가중평균(`weighted_mean`)을 활용한 추론 결과를 모두 저장합니다. (default: `'softvoting'`)

- `save_path`: 추론 결과를 저장할 경로를 지정합니다. (default: `'./prediction'`)

- `tta `: TTA(Test Time Augmentation) 값을 지정합니다. 가령, `tta=2`로 설정할 경우, 추론 과정에서 하나의 이미지를 2개로 Augmentation, 2개의 추론 결과를 앙상블하여 최종적인 추론을 하게 됩니다. (default: `1`)



## Dataset

### I. Split

![data split](https://github.com/iloveslowfood/ImageClassfication/blob/main/etc/data%20split.png?raw=true)

주어진 학습 데이터 중 90%를 학습용 데이터로, 나머지 10%를 검증용 데이터로 활용했습니다. 합리적 검증을 위해 데이터를 이미지 단위가 아닌 사람 단위로 분리했는데, 이는 이미지 단위로 데이터를 분리할 경우 특정 사람의 이미지가 학습용 데이터와 검증용 데이터 모두에 등장해 검증 결과를 신뢰할 수 없는 data leakage 문제가 발생할 수 있기 때문입니다. 또한, 주어진 학습 데이터의 분포가 public/private 데이터의 분포와 같다는 가정 하에, 학습용 데이터와 검증용 데이터의 분포가 같도록 층화추출법을 활용해 주어진 데이터를 분리하였고, 검증용 데이터에는 어떠한 가공도 취하지 않음으로써 검증 결과의 신뢰성을 확보했습니다.

### II. Oversampling

![mixup](https://github.com/iloveslowfood/ImageClassfication/blob/main/etc/mixup.png?raw=true)

주어진 데이터는 18가지 카테고리별 불균형이 존재합니다. 때문에 비교적 부족한 카테고리의 데이터를 오버샘플링한 데이터셋을 추가 구성, 주어진 데이터와 더불어 모델 실험에 활용했습니다.

## Augmentation

### I. Base

CentorCrop 등 가장 일반적인 이미지 가공 방법으로 구성된 Augmentation입니다. ([소스코드 보기](https://github.com/iloveslowfood/ImageClassfication/blob/a14c97f0d2253122a798913fbd29a7bdcb92f128/augmentation.py#L9))

### II. Random

이미지를 임의로 가공하는 방법을 포함한 Augmentation으로, [RandAugment](https://github.com/ildoonet/pytorch-randaugment) 모듈을 활용합니다.([소스코드 보기](https://github.com/iloveslowfood/ImageClassfication/blob/a14c97f0d2253122a798913fbd29a7bdcb92f128/augmentation.py#L102))

### III. TTA

TTA(Test Time Augmentation)에 활용하기 위한 Augmentation으로, Train 단계에서는 ‘Random’ Augmentation과 같은 Augmentation이 진행되고, Inference 단계에서는 `RandomResizedCrop()`의 무작위적 Augmentation을 활용한다는 특징이 있습니다.([소스코드 보기](https://github.com/iloveslowfood/ImageClassfication/blob/a14c97f0d2253122a798913fbd29a7bdcb92f128/augmentation.py#L102))

### IV. Face Crop

픽셀의 분포를 바탕으로 이미지 내 얼굴 부분을 crop하는 Augmentation입니다.([소스코드 보기](https://github.com/iloveslowfood/ImageClassfication/blob/a14c97f0d2253122a798913fbd29a7bdcb92f128/augmentation.py#L102))



## Models

### I. VanillaEfficientNet

![veffi](https://github.com/iloveslowfood/ImageClassfication/blob/main/etc/veffi.png?raw=true)

Pretrained EfficientNet(`'efficientnet-b3'`)을 Backbone으로 하는 간단한 이미지 분류 모델입니다. ([소스코드 보기](https://github.com/iloveslowfood/ImageClassfication/blob/05f60efadc8865b5f76e9503881b5337e5d64313/model.py#L43))

### II. VanillaResNet

![vres](https://github.com/iloveslowfood/ImageClassfication/blob/main/etc/vres.png?raw=true)

Pretrained ResNet(`resnet50`)을 Backbone으로 하는 간단한 이미지 분류 모델입니다. ([소스코드 보기](https://github.com/iloveslowfood/ImageClassfication/blob/05f60efadc8865b5f76e9503881b5337e5d64313/model.py#L70))

### III. MultiClassTHANet

![thanet](https://github.com/iloveslowfood/ImageClassfication/blob/main/etc/thanet.png?raw=true)

Pretrained Image Network와 Attention 아키텍쳐를 활용한 이미지 분류 모델입니다. ([소스코드 보기](https://github.com/iloveslowfood/ImageClassfication/blob/7ef05acccfa04a386a6b98a4e471e8572ea75ff2/model.py#L96))

### IV. MultiLabelTHANet

![thanet_ml](https://github.com/iloveslowfood/ImageClassfication/blob/main/etc/thanet_ml.png?raw=true)

Pretrained Image Network와 Attention 아키텍쳐를 활용한 이미지 분류 모델입니다. ([소스코드 보기](https://github.com/iloveslowfood/ImageClassfication/blob/7ef05acccfa04a386a6b98a4e471e8572ea75ff2/model.py#L171))

### V. THANet

![thanet_3](https://github.com/iloveslowfood/ImageClassfication/blob/main/etc/thanet_3.png?raw=true)

Pretrained Image Network를 활용한 Multi-label 이미지 분류 모델입니다. ([소스코드 보기](https://github.com/iloveslowfood/ImageClassfication/blob/7ef05acccfa04a386a6b98a4e471e8572ea75ff2/model.py#L240))