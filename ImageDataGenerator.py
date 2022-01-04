'''
training data를 늘리기 위한 방법.
tensorflow.keras.preprocessing.image의 ImageDataGenerator를 사용한다.

ImageDataGenerator는 다음과 같은 인자를 받는다.
rescale : 이미지의 픽셀 값을 rescale 한다.
rotation_range  : 이미지를 회전시켜서 랜덤하게 생성한다.
width_shift_range   : width를 이동시킨다.
height_shift_range  : height를 이동시킨다.
shear_range     : 이미지의 수평은 고정시키고, 수직 꼭지점으로 이동시키는 정도.
horizontal_flip : 입력을 수평으로 임의로 뒤집는다.
'''

# training 주소를 찾아서 다음 인자의 작업을 수행한다.
TRAINING_DIR = '/tmp/cats-v-dogs/training'
train_datagen = ImageDataGenerator(rescale=1/255.0, 
                                    rotation_range=20, 
                                    width_shift_range=0.2, 
                                    height_shift_range=0.2, 
                                    shear_range=0.2,
                                    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR, 
                                                    target_size=(300,300), 
                                                    class_mode='binary', 
                                                    batch_size=20)

# testing data의 경우 이미지를 증강할 필요가 없기에, rescale만 실행한다.
VALIDATION_DIR = '/tmp/cats-v-dogs/testing'
validation_datagen = ImageDataGenerator(rescale=1/255.0)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, 
                                                                target_size=(300,300), 
                                                                class_mode='binary', 
                                                                batch_size=20)