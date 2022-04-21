'''

pre-trained된 모델을 사용하면, 학습시간을 줄이고 정확도를 높힐 수 있다.
CNN 부분을 base model 이라고 하고, 분류 부분을 head model이라 하며,
base model의 모델과 weight를 가져오고, head model은 설계해준다.

이 파일에서 base model은 VGG16을 사용하였으며, base model명을 conv_base 라고 하였다.



'''
# 1. base model를 부르고, 바로 합치기
conv_base = keras.applications.VGG16(include_top=False, input_shape=[224, 224, 3]) 
conv_base.trainable = False

model = keras.Sequential()
model.add(conv_base)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(5, activation='softmax'))

# 2. head model을 따로 설계하고, 모델 완성하기
head_model = conv_base.output
head_model = tf.keras.layers.Dense(units=1, activation='sigmoid')(head_model)
model = tf.keras.models.Model(inputs = base_model.input, outputs = head_model)

# 3. 원하는 부분까지 학습시키기 : fine_tune_at 이후 layer를 학습시킨다.
conv_base.trainable = True
fine_tune_at = 100
for layer in conv_base.layers[ 0 : fine_tune_at ] :
    layer.trainable = False
