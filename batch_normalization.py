'''
배치 정규화는 신경망이 깊어질수록 나타나는 Internal Covariate Shift 문제를 해결하기 위해 나온 기법이다.

conv2d_bn은 다음과 같은 인자를 받는다.
x       : conv2d_bn의 입력
filters : 필터의 갯수
num_row : 필터의 row 크기
num_col : 필터의 colunm 크기
padding : 패딩 여부
strides : stride 여부
'''
from tensorflow.keras import layers

def conv2d_bn(x, filters, num_row, num_col, padding= "same", strides=(1, 1)):
    x = layers.Conv2D(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation('relu')(x)
    return x