"""
tensorflow.keras.callbacks의 EarlyStopping을 사용하는 방법.
EarlyStopping은 다음과 같은 인자를 받는다.

monitor   : 모델의 성능을 체크할 때, 기준이 되는 값. 예시) val_loss, val_accuracy
verbose   : 1일 경우 저장되있음이 표시되고, 0일 경우 표시 없이 저장된다.
mode      : auto, min, max. val_accuracy는 높아야 좋은 모델이므로 max,
            val_loss는 낮아야 좋은 모델이므로 min 으로 설정하고, auto는 알아서 설정해준다.
verbose   : 1일 경우 저장되있음이 표시되고, 0일 경우 표시 없이 저장된다.
patience  : patience 값 동안 모델이 개선되지 않는다면, 학습을 중단한다.
"""
from tensorflow.keras.callbacks import EarlyStopping

# val_loss를 기준으로, 10회 모델이 개선되지 않는다면, 학습을 중단하는 코드
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# early stopping은 model.fit에 callbacks의 인자로 들어간다.
history = model.fit(X_train, y_train, 
                      callbacks=[checkpoint, early_stopping], 
                      batch_size=50, 
                      validation_data = (X_test,y_test), 
                      epochs = 100)