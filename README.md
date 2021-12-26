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

