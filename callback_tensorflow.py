'''
tensorflow.keras.callbacks의 ModelCheckpoint 사용하는 방법.
ModelCheckpoint는 다음과 같은 인자를 받는다.
filepath  : 모델을 저장할 경로.
monitor   : 모델을 저장할 때, 기준이 되는 값. 예시) val_loss, val_accuracy
verbose   : 1일 경우 저장되있음이 표시되고, 0일 경우 표시 없이 저장된다.
save_best_only  : True와 False. True일 시 monitor로 선정된 값중 가장 좋은 값을 저장하고,
                  False일 시 매 에폭마다 모델이 filepath{epoch}로 저장된다.
save_weights_only : True인 경우 weights만 저장. False인 경우 weights와 model 모두 저장.
mode      : auto, min, max. val_accuracy는 높아야 좋은 모델이므로 max,
            val_loss는 낮아야 좋은 모델이므로 min 으로 설정하고, auto는 알아서 설정해준다.
save_freq : epoch를 사용할 경우 매 에폭마다 모델을 저장한다.
            integer를 사용할 경우, 입력 숫자만큼 배치가 진행되며 모델이 저장된다.
'''
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# 모델이 저장될 주소. 없다면 생성한다.
model_dir = '/content/drive/MyDrive/Colab Notebooks/deep_running'
if not os.path.exists(model_dir) :
  os.mkdir(model_dir)
model_path = model_dir + "/classification_224.model"

# val_loss를 기준으로 가장 좋은 모델만 저장하는 checkpoint
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)

# checkpoint는 model.fit에 callbacks의 인자로 들어간다.
history = model.fit(X_train, y_train, 
                      callbacks=[checkpoint, early_stopping], 
                      batch_size=50, 
                      validation_data = (X_test,y_test), 
                      epochs = 100)