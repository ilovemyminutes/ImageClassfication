# Image Classification

향후 모든 기록은 [이곳](https://www.notion.so/iloveslowfood/Stage-2-Image-Classification-58dbfca2e1ef4e36b8de6790b403ccba)에 업데이트됩니다.

## Task Description

- ***Problem Type.\*** Classification - 마스크/성별/연령대에 따른 18개 클래스
- ***Metric.\*** Macro F1 Score
- ***Data.\*** 한 명당 7장(마스크 착용x1, 미착용x1, 불완전 착용x5) ,총 *2*,700명의 이미지. 한 사람당 384x512



## Performances

- F1 0.7706, Private LB 0.7604

- Configuration

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

  

### 2021.03.31

###### *PLAN: 모델 파이프라인 재구성, 데이터 Augmentation, Train/Valid/Test 재구성*

> ***Main Issue: 버그 해결 - ImageFolder*** 

![image-20210331190359100](https://github.com/iloveslowfood/iloveTIL/blob/main/boostcamp_ai/etc/images/PStage%20-%2001.%20Image%20Classification/board.png?raw=true)

- 세상에나 마상에나, 문제를 해결했다. 별다른 Augmentation 없이 아주 간단한 모델을 돌린지라 성능은 아직 부족하나, 존재했던 버그를 해결함으로써 내가 구축한 파이프라인에 신뢰감을 갖게 되었다. 이제 성능 끌어올릴 일만 남았군!

- 문제는 `ImageFolder` 클래스때문이었다. `ImageFolder` 는 이미지 데이터 활용 모델 학습을 더욱 편리하게 할 수 있도록 하는 `torchvision`의 클래스이다. 이러한 편리성은 클래스별로 폴더를 만들어 데이터를 넣어두면, 별도로 레이블링을 진행하지 않아도, 폴더별 클래스를 매기기 때문인데,  이러한 간편함이 문제를 불러일으켰다.

- 문제는 **의도대로 레이블링이 진행되지 않았다는데서 발생**했다. 감사하게도 이정환 캠퍼님께서 이 부분에 대한 코멘트를 주셨는데, 실제로 확인해보니 다음과 같이 레이블링이 이루어지고 있었다.

  ```python
  {'0': 0,
   '1': 1,
   '10': 2,
   '11': 3,
   '12': 4,
   '13': 5,
   '14': 6,
   '15': 7,
   '16': 8,
   '17': 9,
   '2': 10,
   '3': 11,
   '4': 12,
   '5': 13,
   '6': 14,
   '7': 15,
   '8': 16,
   '9': 17,
   }
  ```

- 관련 소스코드를 살펴보면서 문제를 해결하게 되었는데, 해결 과정은 [토론 게시판](http://boostcamp.stages.ai/competitions/1/discussion/post/18)을 통해 확인할 수 있다.



> ***Review***

- 모델 train phase가 잘못된건가, 데이터를 잘못 처리했나, 모델 구축이 잘못되었나, ... 참 여러 방면으로 고민되었고 혼란스러웠다.
- 처음에는 되게 간단한 부분처럼 보였기에, 다른 캠퍼분들께 공유하지 않았는데, 공유하고 나니 이렇게 일찍이 문제가 해결되었구나. 진작 공유할 걸 그랬다. 앞으로 다른 캠퍼분들이 문제가 생기면 적극적으로 고민해봐야겠다.
- 이제 성능 높일 일만 남았다! 데이터 확실히 탐색하고 그리 원하던 `AttentionNet(가칭)`을 빨리 구축해보고 싶다.
- [김기민 캠퍼분께서 언급하신 데이터 split에 대한 부분](http://boostcamp.stages.ai/competitions/1/discussion/post/17)은 지속적으로 염두에 둬야할 부분으로 보인다. Robust한 모델을 구축하기 위해 필요한 요소다.



### 2021.03.30

###### *PLAN:* eval 이미지 데이터에 대한 EDA 및 augmentation 방법 고안

> ***Main Issue: No Generalization***

- 모델 일반화가 떨어지는 것으로 보인다. 아, 있었는데? 아니 그냥 없어요😂

- `VanillaResNet` 모델은 26%로 성능이 좋지 않았다. 예상했던 것보다 더.

- 정확히는, **Train/Validation phase에서는 80%대의 정확도**를 보였으나, (매우 당황스럽게도) **리더보드에는 26% 가량**의 정확도가 찍혔다.

- 학습이 잘못된 것인지 코드를 살펴보았으나, train과 validation 데이터셋에 대한 inference는 80% 정확도를 납득할 만한 추론이 진행되는 것으로 볼 때, 모델의 generalization이 떨어진 것으로 보인다.

- !!!근데, 조교님과의 피드백 결과, train phase에서 뭔가 잘못되었을 수 있겠다는 판단을 하게 되었다. 코드를 더 살펴보자
  - 우선, 학습 동안의 validation 코드와 inference 동안의 코드를 같은 방식으로 동작하도록 수정했음
  - 현 피어세션과 전 피어세션 피어분들과 이야기 나누다보니, ***각 이미지에 클래스 할당을 잘못했다***는 생각이 들었다. 모델을 학습하여 제출해본 피어분들은 아직까지 pretrain된 모델을 활용해서, 내가 구성한 그것과 같았다.
  - 내가 구성한 모델도 Train, Validation 단계에서는 성능이 일관적으로 나오는 반면, 리더보드에 상이한 점수가 나오는 것을 보면, 잘못된 카테고리로 모델이 학습된 것으로 보인다.
  
- ***에러 발견***

  - pretrained model을 backbone으로 불러와 freeze하는 과정에서 `param.requires_grad = False`로 설정하는 부분이 있는데, 이 부분을 잘못 작성했다.

    ```python
    class VanillaEfficientNet(nn.Module):
        def __init__(self, freeze: bool = True):
            super(VanillaEfficientNet, self).__init__()
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
            if freeze:
                self._freeze()
            self.linear = nn.Linear(in_features=1000, out_features=18)
    
        def forward(self, x):
            output = self.efficientnet(x)
            output = self.linear(output)
            return output
    
        def _freeze(self):
            for param in self.efficientnet.parameters():
                param.requires_grad = False # 기존 작성 코드: param.require_grad = False
    ```

    

> ***EDA***

- 제출 결과에 따르면, train 데이터와 eval 데이터는 그 분포에 괴리가 다소 있는 듯 보인다.
- 인물 사진은 그 뒷배경과 함께 담겨있을 수밖에 없는데, 러프하게 살펴보니 어떤 인물의 사진은 멀리서 찍기도, 어떤 인물은 가까이서 찍기도하며, 장소에 따라 뒷배경의 색상, 밝기 등이 다르다.
- 김태진님의 강의에 따라 ***모델 학습에 도움이 될 것 같은 방향***으로 EDA를 해볼 참이다.
  - 가장 우선적으로 내가 하고싶은 것은, ***얼굴 부분만을 크롭하는 방법을 고안***하는 것이다.

> ***Model: New Architecture***

- 어제 생각했던 `AttentionNet(가칭)`에 대해 조금 더 구체화해보았다.

- 모델에 반영하고자 한 것은 '마스크', '성별', '나이대'의 3가지 task를 end-to-end 학습이 가능하도록 하는 것

  - *왜?* 우선적으로, Pretrained 모델을 활용하는 것이 효율적으로 높은 성능을 얻을 수 있는데, 이를 더욱 효과적으로 활용할 수 있는 방법은 없을지 고민하는 것에서 시작되었다. 카테고리가 마스크 상태, 성별, 나이대의 3가지 요소를 고려하여 구성되어있는데, 이 3가지 task를 독립적인 사건으로 가정한다면, 3가지 요소에 대한 예측을 개별적으로 수행하면 더욱 높은 정확도가 나올 것으로 생각했다.

  - 첫째로 떠올랐던 것: 3가지 task를 수행한 뒤 절충하는 과정을 거치는 모델

    ![model architecture2](https://github.com/iloveslowfood/iloveTIL/blob/main/boostcamp_ai/etc/images/PStage%20-%2001.%20Image%20Classification/model%20architecture2.png?raw=true)

    - RCNN Family의 모델로부터 아이디어를 얻었던 방법이다. Fast RCNN 모델에서 바운딩박스와 카테고리 예측을 별도로 수행하는 것처럼, pretrain된 ImageNet을 활용하여 1차적으로 feature map을 얻은 뒤, 3가지 task를 개별적으로 수행한다.
    - (한계)*어떻게 학습할건데?*: - 이러한 모델은 ***각각의 task로부터 얻은 loss값을 어떻게 통합할 것이며, 어떻게 weight를 업데이트하는 것이 좋을지를*** 판단하는 것이 중요해보였다. 가장 간단한 방법은 단순 평균을 사용하는 방법이겠으나, *'Loss를 더욱 interactive하게 가중합하는 방법은 없을까?'*라는 의구심이 들었다.
      - 직관적으로는 loss 각각에 대해 trainable한 가중치를 마련하면 좋겠다고 생각할 수 있는데, loss는 기본적으로 모델이 추론해낸 값과 ground truth 간의 분포 비교를 통해 측정되는 것이다보니, ***loss 자체에 trainable한 가중치를 도입한다는 것은 어불성설***이다.
      - 결국 단순 평균을 제외하고는 명확한 수가 떠오르지 않았는데, 한편 이로부터 attention을 도입하는 것은 어떨지에 대한 호기심이 생겼다.

  - Attention을 써먹으면 더 좋지 않을까?

    ![model architecture3](https://github.com/iloveslowfood/iloveTIL/blob/main/boostcamp_ai/etc/images/PStage%20-%2001.%20Image%20Classification/model%20architecture3.png?raw=true)

    - Loss에 현명한 가중치를 부여하는 방법을 고민하다가 고안하게된 모델 구조
    - 일반적으로 end-to-end 학습이 더 높은 성능을 보이는 것으로 알려져있고, 이 competition의 문제로서는 인공신경망이 '알아서 마스크 상태, 성별, 나이대를 적절히 고려해주었으면 좋겠다' 정도로 생각해볼 수있다.
    - fully connected 레이어를 통해 각 Task별 hidden state를 출력한 뒤, 이를 바탕으로 attention 텐서를 생성
    - 구한 attention value를 바탕으로 기존의 3가지 hidden state를 가중한 뒤 이러한 hidden state를 하나의 layer로 구성하여 최종 예측
    - 즉, 내가 모델에 바라고 있는 것은, 'task 각각을 초기에는 independent하게 파악한 뒤, 파악한 패턴을 적절히 절충하여 클래스를 예측하는 것'이다.
    - 논리적으로 맞지 않는 부분도 많은데, 우선 호기심이 생겨버려서 만들어보지 않을 수가 없게되어버렸다
    - [Transformer 구현](https://github.com/iloveslowfood/iloveTIL/tree/main/pytorch/transformer)해보길 잘했네!

> ***References***

- 특강에서 김상훈 캐글 그랜드마스터께서 [레퍼런스로 올려주신 코드](https://github.com/lime-robot/categories-prediction)를 살펴보았는데, 깜짝 놀랐고, 앞으로 좋은 레퍼런스가 될 수 있다고 생각되었다.
- 코드는 정말 정갈하고, 눈으로 훑어 내려가더라도 대략적으로 어떤 task를 수행하는 부분인지 명확하게 파악하는 것이 가능했다.
- [오혜린님이 토론](http://boostcamp.stages.ai/competitions/1/discussion/post/6)에서 올려주신 EDA  덕분에 데이터에 대한 전반적인 컨셉을 파악할 수 있었다. 나도 빨리 디버깅을 마치고 내 의견을 공유해보고 싶다!!!
- 나의 코드를 믿을 수 있으려면, 코드가 정갈해야 하고, 깊이 알아갈 수록 심오하나 얕게 보아도 이해가 되어야 한다.

### 2021.03.29

###### *PLAN.* 김상훈님의 조언에 따라 베이스라인을 먼저 구축한 뒤, 이에 뒤따라 EDA를 진행

- *Garbage in, garbage out.* 모델 아키텍쳐보다 확실한 데이터셋을 우선적으로 구성하는 것이 더 좋을듯

> ***Model***

- `VanillaResNet`
  - Pretrained ResNet을 활용한 기본적인 분류기
  - 앞쪽의 block layer들은 freeze하고 마지막 fc 레이어만을 update하여 학습

- `AttetionNet(가칭)`

  - Faster-RCNN과  GoogLeNet의 구조로부터 ideation
  - 성별, 마스크 상태, 연령대를 각기 다른 task로 나누어 예측
  - Pretrained ImageNet을 활용해 이미지로부터 feature를 추출한 뒤, 추출한 feature로부터 3가지의 task를 수행(함을 가정)

  ![model architecture](https://github.com/iloveslowfood/iloveTIL/blob/main/boostcamp_ai/etc/images/PStage%20-%2001.%20Image%20Classification/model%20architecture.png?raw=true)

> ***Train Configuration***

- 모델 간 합리적 비교를 위해 학습 환경을 통일

- `Epoch`: 5

- `Optimizer`: Adam

- `Loss Function`: CrossEntropyLoss

- `Batch Size`: 128

- `Transform Type`: `base` (향후 추가)

  - `base`: 

    - Train

      ```python
      transforms.CenterCrop((384, 384)),
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ```

    - Test

      ```python
      transforms.CenterCrop((384, 384),
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ```

- `Learning Rate`: 5e-4 / 25e-5
- `Seed`: 42

> ***Review***

- 확실히 학습 파이프라인 구축 속도는 빨라진 것 같다. 다만, 사용하는 함수만 사용하는 경향이 짙어질 휘업