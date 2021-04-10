# Image Classification

Daily Contributions는 [이곳](https://www.notion.so/iloveslowfood/Stage-2-Image-Classification-58dbfca2e1ef4e36b8de6790b403ccba)에 업데이트됩니다.

## Task Description

- ***Problem Type.*** Classification - 마스크/성별/연령대에 따른 18개 클래스
- ***Metric.*** Macro F1 Score
- ***Data.*** 한 명당 7장(마스크 착용x1, 미착용x1, 불완전 착용x5) ,총 *2*,700명의 이미지. 한 사람당 384x512



## Performances

##### Public LB F1 0.7706, Private LB 0.7604

##### Configuration

```python
batch_size=32
epochs=모델 별 상이
loss_type='labelsmoothingLoss'
lr=0.001
lr_scheduler='cosine'
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

- 18가지 카테고리를 분류하는 Main Task 이외에 마스크 상태, 연령대(classification) 및 연령(regression), 성별의 4가지 sub task에 대한 학습을 모두 지원하며, K-Fold CV 학습 또한 가능합니다. 조정 가능한 argument는 다음과 같습니다.

- `task` : main task(*main*), 마스크 상태(*mask*), 연령대(*ageg*), 연령(*age*), 성별(*gender*)의 5가지 task에 대한 학습이 가능합니다. (default: *main*)

- `model_type`: 학습할 모델을 선택합니다. 지원하는 모델 아키텍쳐는 `VanillaEfficientNet`, `VanillaResNet`, `MultiLabelTHANet`, `MultiClassTHANet_MK1`, `THANet_MK1`, `THANet_MK2`이 있습니다. 

- `load_state_dict`: 저장된 모델을 불러와 학습할 경우 저장된 파일의 경로를 입력합니다. 저장된 파라미터와 `model_type`이 일치해야 합니다.

- `train/valid_root`: 학습용 데이터와 검증용 데이터의 경로를 입력합니다.

- `transform_type`: Augmentation에 활용할 Transform 종류를 입력합니다. *Base*, *Random*, *TTA*, *Face Crop*을 지원합니다.

- `age_filter`: 일정 나이 이상인 인물의 연령대를 60대 이상 연령대로 강제 변경합니다. 50대 후반의 인물과 60대 인물의 사진은 분간하기가 어려워 예측 성능에 지장을 주는 경우가 있었기 때문에 고안한 argument입니다. 가령,  age_filter를 58로 설정할 경우, 58세 이상인 인물 모두가 '60대 이상 연령대' 범주로 강제 변경됩니다.

- `epochs`: 에폭을 설정합니다. (default: 30)

- `cv`: KFold CV를 활용할 경우 사용하는 argument로, Fold 수를 설정합니다. 입력하지 않거나 1을 입력할 경우 단일 폴드로 학습, 즉 KFold CV가 진행되지 않습니다. (default: 1)

- `batch_size`: 배치 사이즈를 설정합니다. (default: 32)

- `optim_type`: 최적화 함수를 설정합니다. *Adam*, *AdamP*, *SGD*, *Momentum*을 지원합니다. (default: `adam`)

- `loss_type`: 손실 함수를 설정합니다. *Label Smoothing Loss*, *Focal Loss*, *Cross Entropy Loss*, *MSE Loss*, *Smooth L1 Loss*를 지원합니다. (default: `labelsmootingloss`)

- `lr`: Learning Rate를 설정합니다. (default: `0.005`)

- `lr_scheduler`: LR 스케줄러를 설정합니다. *Cosine Annealing LR Decay*를 지원합니다. (default: `cosine`)

  

### Inference Phase

##### ***Singular Model Inference***

```python
>>> python submit_singular --task 'main' --model-type 'VanillaEfficientNet' --transform-type 'random'
```

- `task` : 메인 task(`'main'`), 마스크 상태(`'mask'`), 연령대(`'ageg'`), 연령(`'age'`), 성별(`'gender'`)의 5가지 task에 대한 추론이 가능합니다. (default: `'main'`)

- `model_type` : 불러올 모델 아키텍쳐를 선택합니다. 지원하는 모델 아키텍쳐는 ***VanillaEfficientNet***(`'VanillaEfficientNet'`), ***VanillaResNet***(`'VanillaResNet'`), MultiLabelTHANe (`'MultiLabelTHANet'`), ***MultiClassTHANet_MK1***(`'MultiClassTHANet_MK1'`), THANet_MK1(`'THANet_MK1'`), THANet_MK2(`'THANet_MK2'`)이 있습니다. (default: `'VanillaEfficientNet'`)

- `load_state_dict` : 추론에 활용할 사전 학습된 파라미터 파일의 경로를 설정합니다. 모델 아키텍쳐에 맞는 파라미터 파일을 불러와야 정상 작동합니다.

- `transform_type`: Augmentation에 활용할 Transform 종류를 입력합니다. ***Base***(`'base'`), ***Random***(`'random'`), ***TTA***(`'tta'`), ***Face Crop***(`'facecrop'`)을 지원하며, 각 transform은 다음과 같습니다. (default: ***Base***(`'base'`))

- `data_root`: 추론할 데이터의 경로를 입력합니다. (default: `./input/data/images`)

- `save_path`: 추론 결과를 저장할 경로를 입력합니다. 추론 결과는 ImageID와 ans의 두 컬럼을 포함한 csv 파일 형태로 저장됩니다. (default: `'./prediction'`)

  

##### ***Ensemble Inference***

```python
>>> python submit_ensemble --task 'main' --root './saved_ensemble_models' --transform-type --method 'soft' --top-k 3 --tta 2
```

- `task` : 메인 task(`'main'`), 마스크 상태(`'mask'`), 연령대(`'ageg'`), 연령(`'age'`), 성별(`'gender'`)의 5가지 task에 대한 추론이 가능합니다. (default: `'main'`)

- `root` : 앙상블할 모델이 저장된 폴더 경로를 설정합니다. 현재 KFold 기반의 앙상블만이 지원되기 때문에, 입력한 경로는 다음과 같은 디렉토리 구조를 가져야 합니다. KFold의 최상위 폴더의 이름을 기준으로 모델 아키텍쳐를 불러오기 때문에, 최상위 폴더에는 모델명이 반드시 기재되어야 합니다. 

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





