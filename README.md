# ai_skills

ai_skills 에는 딥러닝 모델을 학습시킴에 있어서 유용한 코드들을 모아두었습니다.     
- 학습 데이터셋 저장   
- callback   
- early stopping    



## 1. 학습 데이터셋 저장 : save_files.py
save_flies.py에는 이미지 데이터를 모델에 맞게 resize 하고,   
test셋과 train셋으로 분리하여 랜덤하게 섞어준 전체 데이터셋을 저장하는 코드입니다.   
   
모델 학습 마다 데이터를 읽을 필요 없이, 저장된 데이터셋을 사용하면 쉽게 load 할 수 있습니다.

## 2. 학습된 모델 저장(ModelCheckpoint 사용) : callback_tensorflow.py
callback_tensorflow.py에는 학습된 모델을 저장하는 callback 코드가 들어있습니다.
callback은 tensorflow.keras.callbacks의 ModelCheckpoint를 사용했습니다.

ModelCheckpoint가 받는 인자를 설명하였고,
model.fit을 할 때, callbacks가 인자로 들어가는 코드도 넣었습니다.

## 3. epochs 달성 전 학습 종료(EarlyStopping 사용) : Early_stopping.py
Early_stopping.py에는 모델이 일정 이상 개선되지 않을 때, 더이상 모델을 학습시키지 않고 종료하는 코드가 들어있습니다.

모델을 학습 시키는 경우, 과적합으로 인해 overfitting 문제가 일어납니다.
overfitting은 train 데이터에 대해 정확도가 올라가지만, test나 validation 데이터에 대해 정확도가 오르지 않는 경우입니다.
따라서 이러한 상황이 학습이 진행됨에 따라 나타나는 경우, 모두 학습하기 전에 학습을 멈추는 기법이 Early stopping 입니다.

EarlyStopping이 받는 인자를 설명하였고,
model.fit을 할 때, EarlyStopping이 인자로 들어가는 코드도 넣었습니다.

## 4. 정확도가 일정 이상일 때 학습 종료 : callback_myCallback.py
callback_myCallback.py에는 모델의 정확도가 일정 이상일 때, 학습을 종료하는 코드입니다.
tensorflow.keras.callbacks의 Callback을 사용했습니다.

자신이 원하는 정확도에서 학습을 종료시킬 수 있습니다.
myCallback 이라는 함수를 만들고, val_accuracy가 설정한 값 이상이면, 학습을 멈추는 코드입니다.

class로 myCallback을 선언하였고, 그 안에 val_accuracy가 나올떄마다 판단하는 코드를 넣었습니다.

## 5. 이미지 증강 : ImageDataGenerator.py
ImageDataGenerator.py에는 training data가 부족하거나, 늘리고 싶을 때, 사용하는 기법입니다.
이미지를 회전시키거나, 선대칭, 수평 및 수직 이동 시킴으로써 training data를 늘려줍니다.

## 6. 배치 정규화 : batch_normalization.py
배치 정규화는 신경망이 깊어질수록 나타나는 Internal Covariate Shift 문제를 해결하기 위해 나온 기법입니다.
Internal Covariate Shift는 레이어를 통과할 때마다,
Covariate Shift가 일어나면서 입력의 분포가 약간식 변하는 현상입니다.
레이어 마다 정규화 하는 레이어를 두어, 변형된 분포가 나오지 않도록 조절합니다.

미니 배치의 평균과 분산을 이용해서 정규화 한 뒤에,
scale 및 shift를 감마, 베타를 통해 실행하고, 이는 역전파로 학습시킬 수 있습니다.
이는 비선형 성질을 유지하면서 학습 될 수 있게 해줍니다.

참고 자료 : https://eehoeskrap.tistory.com/430

## 7. 기학습된 모델 사용하기 : fine_tuning.py
기학습된 모델은 데이터 학습시간을 줄이기 위해 사용합니다.

모델은 크게 필터인 CNN과 분류인 FCN으로 이루어져 있습니다.
CNN 부분을 base model 이라고 하며, FCN 부분을 head model 이라고 합니다.
기학습된 base model을 사용하면(trainalbe = False) 변화시킬 weight가 head model의 그것 뿐입니다.
따라서 학습시간을 줄일 수 있고, 빠르게 원하는 목표치에 도달할 수 있습니다.

이 파일에선 학습된 모델에 이어서 모델을 작성하는 방법과,
head model을 생성해서 model로 붙히는 방식 2가지를 소개했습니다.

또한, base model의 원하는 부분까지 기학습된 weight 값을 가져오고,
나머지 layer를 학습시키는 코드를 넣었습니다.
