# ai_skills

ai_skills 에는 딥러닝 모델을 학습시킴에 있어서 유용한 코드들을 모아두었습니다.     
- 학습 데이터셋 저장   
- callback   
- early stopping    



## 1. 학습 데이터셋 저장 : save_files
save_flies.py에는 이미지 데이터를 모델에 맞게 resize 하고,   
test셋과 train셋으로 분리하여 랜덤하게 섞어준 전체 데이터셋을 저장하는 코드입니다.   
   
모델 학습 마다 데이터를 읽을 필요 없이, 저장된 데이터셋을 사용하면 쉽게 load 할 수 있습니다.
