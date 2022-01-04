"""
이미지를 reshape 한 형태로 저장하는 코드.
모델이 달라지더라도, save 파일을 불러와서 학습 및 테스트 시킬 수 있다.

img_dir : 이미지 폴더가 들어있는 주소.
categories : img_dir 안에 카테고리마다 이미지가 각각 다른 폴더에 들어있다.

image_w : resize 할 이미지의 w 크기
image_h : resize 할 이미지의 h 크기

X : resize 된 이미지의 각 픽셀값이 numpy 형태로 저장됨.
y : 이미지의 카테고리 값

"""

from PIL import Image
import os, glob, sys
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '/content/drive/MyDrive/Colab Notebooks/raw-img'
categories = ['dog','cat']
np_classes = len(categories)

image_w = 100
image_h = 100

X=[]
y=[]

# 파일을 리스트 형태로 저장한다.
for idx, messy_desk in enumerate(categories) : 

    img_dir_detail = img_dir + "/" + messy_desk
    files = glob.glob(img_dir_detail + "/*") 

    # 폴더 하나씩 접근하여 파일을 열고, resize를 진행 후 X와 y에 저장한다.
    for i, f in enumerate(files) :
        try :
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img)

            # X,y에 리스트 형태로 저장
            # y는 카테고리 순서 이므로 idx 값으로 넣는다.
            X.append(data)
            y.append(idx)

            # 100번째 파일마다 실행 과정을 출력한다.
            if i % 100 == 0 :
                print(messy_desk, ":", f)
      
        except :
            print(messy_desk, str(i) + " 번째에서 에러", f)

X = np.array(X)
y = np.array(y)

# train set 0.9, test set 0.1
# 데이터 셋이 셔플된 상태로 저장된다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# 전처리 된 파일을 저장한다 : check point
save_files = (X_train, X_test, y_train, y_test)
np.save("/content/drive/MyDrive/Colab Notebooks/practice/binary_Image_data.npy", save_files)