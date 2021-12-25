'''
모델을 일정 이상 정확도가 되면 학습을 중단하는 방법.
tensorflow.keras.callbacks의 ModelCheckpoint 사용한다.

logs에 'accuracy'를 가져와서, 0.99 이상이면, stop_training을 한다.
'''
import tensorflow as tf

# 모델이 99퍼센트 이상의 정확도를 가질 때, 학습 멈추기.
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if logs.get('val_accuracy') > 0.99 :
            print('\n 정확도가 99% 이상이면, 학습을 멈춘다.')
            self.model.stop_training = True

# 변수로 만들어줘야 cpu가 인식할 수 있다.
my_cb = myCallback()

# my_cb는 model.fit에 callbacks의 인자로 들어간다.
history = model.fit(X_train, y_train, epochs = 30, callbacks = [my_cb])