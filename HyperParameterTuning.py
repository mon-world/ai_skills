'''
하이퍼 파라미터 튜닝
하이퍼 파라미터를 스스로 변화시켜 가며, 최적의 하이퍼 파리미터를 찾는 코드.

이 코드에선 MNIST 데이터셋을 활용하였다.

'''
import tensorflow.keras as keras
import numpy as np
import kerastuner as kt         # keras-tuner


def optimize_hyper_parameter(tuner):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    x_train = x_train / 255
    x_test = x_test / 255

    tuner.search(x_train, y_train, epochs=10, batch_size=100, verbose=2, validation_split=0.2)

    print('search space' + '-' * 30)
    tuner.search_space_summary()

    print('result' + '-' * 30)
    tuner.results_summary()

    print('=' * 50)

    best_hps = tuner.get_best_hyperparameters(num_trials=3)
    print(best_hps[0])
    print(best_hps[0].get('unit1'))
    print(best_hps[0].get('lr'))

    models = tuner.get_best_models(num_models=3)
    model = models[0]

    model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2, validation_split=0.2)
    print(model.evaluate(x_test, y_test, verbose=0))

    # for model in models:
    #     model.fit(x_train, y_train, epochs=10, batch_size=100, verbose=2, validation_split=0.2)
    #     print(model.evaluate(x_test, y_test, verbose=0))


def model_builder(hp):
    hp_unit1 = hp.Int('unit1', min_value=128, max_value=512, step=128)
    hp_lr = hp.Choice('lr', [0.1, 0.01, 0.001])

    model = keras.Sequential()
    model.add(keras.layers.InputLayer([784]))
    model.add(keras.layers.Dense(hp_unit1, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(hp_lr),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    return model


tuner_bayesian = kt.BayesianOptimization(
    model_builder,
    objective='val_loss',
    max_trials=5,
    directory='keras_tuner/bayesian',
    project_name='mnist'
)

tuner_random = kt.RandomSearch(
    model_builder,
    objective='val_acc',
    max_trials=5,
    directory='keras_tuner/random',
    project_name='mnist'
)

# objective    : 최적화할 하이퍼 모델. 이것을 기준으로 최대치를 산출한다.
# max_epochs   : 최대 실행할 epoch
# directory    : 저장 경로
# project_name : 저장 이름
tuner_hyperband = kt.Hyperband(
    model_builder,
    objective='val_loss',
    max_epochs=5,
    directory='keras_tuner/hyperband',
    project_name='mnist'
)

# optimize_hyper_parameter(tuner_bayesian)
# optimize_hyper_parameter(tuner_random)
optimize_hyper_parameter(tuner_hyperband)
















