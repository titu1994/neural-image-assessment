from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD
from keras_tqdm import TQDMCallback
import numpy as np
import time
import bcolz
import joblib
from tqdm import tqdm

from ava_dataset import AvaDataset
from clr_callback import CyclicLR
from config import *
from image_preprocessing import ImageDataGenerator, randomCropFlips, \
    centerCrop224
from nasnet_model import *
# from quick_model import *
# from incept_resnet_model import *
from tensorboard_batch import TensorBoardBatch
from utils.score_utils import srcc

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(
        K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)


def calc_srcc(model, gen, test_size, batch_size):
    y_test = []
    y_pred = []

    for i in tqdm(range(test_size // batch_size)):
        batch = next(gen)
        y_test.append(batch[1])
        y_pred.append(model.predict_on_batch(batch[0]))
    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)
    rho = srcc(y_test, y_pred)
    print("srcc = {}".format(rho))


def lr_schedule(epoch):
    lr = 0.0003
    if epoch > 3:
        lr = 0.0001
    if epoch > 8:
        lr = 0.00005
    if epoch > 11:
        lr = 0.00001
    return lr


def train_top_layers(nima_model, dataset, imggen):
    batch_size = 256
    print("generating features")
    base_model = nima_model.base_model
    nb_train_samples = len(dataset.train_image_paths)
    optimizer = Adam(lr=1e-4)
    for layer in base_model.layers:
        layer.trainable = False
    base_model.compile(optimizer, loss=earth_mover_loss)

    def gen_features():
        scores = []
        features = bcolz.carray(np.empty((0,) + shp[1:]),
                                rootdir=bc_path_features,
                                chunklen=16, mode='w')
        gen = imggen.flow_from_filelist(dataset.train_image_paths,
                                        dataset.train_scores,
                                        shuffle=True, batch_size=batch_size,
                                        image_size=PRE_CROP_IMAGE_SIZE,
                                        cropped_image_size=IMAGE_SIZE)
        for i in tqdm(range(nb_train_samples // (batch_size))):
            # for i,batch in tqdm(enumerate(gen)):
            batch = gen.next()
            features.append(base_model.predict(batch[0]))
            # features.append(base_model2.predict(batch[0]))
            scores.append(batch[1])
            if (i % 100 == 99): features.flush()
        features.flush()
        scores = np.concatenate(scores)
        joblib.dump(scores, scores_path)

    shp = base_model.output_shape
    if not bc_path_features.exists() or not scores_path.exists():
        gen_features()

    scores = joblib.load(scores_path)
    features = bcolz.open(bc_path_features)
    features = np.array(features, dtype=np.float16)

    # train model consisting only of top layers
    ft_input = Input(shape=shp[1:])
    x = Dropout(0.75)(ft_input)
    x = Dense(10, activation='softmax', name='toplayer')(x)
    model_top = Model(ft_input, x)

    checkpoint_top = ModelCheckpoint(weights_top_file,
                                     monitor='val_loss', verbose=1,
                                     save_weights_only=True,
                                     save_best_only=True,
                                     mode='min')
    tensorboard = TensorBoardBatch(log_dir=log_dir)
    optimizer = Adam(lr=0.0003, decay=0.005)
    clr = CyclicLR(base_lr=0.0001, max_lr=0.003,
                   step_size=2 * (len(dataset.train_scores) // batch_size),
                   mode='triangular')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15,
                                   verbose=0, mode='auto')
    callbacks = [clr, early_stopping, checkpoint_top, tensorboard,
                 TQDMCallback()]
    model_top.compile(optimizer, loss=earth_mover_loss)

    print("training top layers")
    model_top.fit(x=features, y=scores, batch_size=128, epochs=40,
                  verbose=0, callbacks=callbacks, validation_split=0.9,
                  shuffle=True)

    nima_model.model.load_weights(weights_top_file, by_name=True)


def main():
    dataset = AvaDataset(dataset_path=dataset_path,
                         base_images_path=base_images_path)

    nima_model = NimaModel()
    model = nima_model.model
    batch_size = 64

    # set up image data generators
    imggen = ImageDataGenerator(preprocessing_function=randomCropFlips())
    trn_gen = imggen.flow_from_filelist(dataset.train_image_paths,
                                        dataset.train_scores,
                                        shuffle=True, batch_size=batch_size,
                                        image_size=PRE_CROP_IMAGE_SIZE,
                                        cropped_image_size=IMAGE_SIZE)

    gen_cent = ImageDataGenerator(preprocessing_function=centerCrop224())
    val_gen = gen_cent.flow_from_filelist(dataset.test_image_paths,
                                          dataset.test_scores,
                                          shuffle=False,
                                          batch_size=batch_size,
                                          image_size=PRE_CROP_IMAGE_SIZE,
                                          cropped_image_size=IMAGE_SIZE)

    tensorboard = TensorBoardBatch(write_graph=False, log_dir="logs/{}".format(
        time.strftime("%Y%m%d-%H%M%S")))
    # scheduler = LearningRateScheduler(lr_schedule)

    if weights_file.exists():
        print("loading weights")
        model.load_weights(weights_file)
    else:
        train_top_layers(nima_model=nima_model, dataset=dataset, imggen=imggen)

    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', verbose=1,
                                 save_weights_only=True, save_best_only=True,
                                 mode='min')
    checkpoint_epoch = ModelCheckpoint(weights_epoch_file, monitor='val_loss',
                                       verbose=1,
                                       save_weights_only=True, mode='min')
    epochs = 25
    optimizer = SGD(lr=0.0003, momentum=0.9, nesterov=True)

    clr = CyclicLR(base_lr=0.000001, max_lr=0.003,
                   step_size=(len(dataset.train_scores) // batch_size),
                   mode='triangular')
    callbacks = [clr, checkpoint, checkpoint_epoch, tensorboard,
                 TQDMCallback()]

    # start training
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer, loss=earth_mover_loss)
    print("training whole model")
    model.fit_generator(trn_gen,
                        steps_per_epoch=(
                                    len(dataset.train_scores) // batch_size),
                        epochs=epochs, verbose=0, callbacks=callbacks,
                        validation_data=val_gen,
                        validation_steps=(dataset.test_size // batch_size),
                        workers=16,
                        initial_epoch=0
                        )
    print("calculating spearman's rank correlation coefficient")
    calc_srcc(model=model, gen=val_gen, test_size=dataset.test_size,
              batch_size=batch_size)


if __name__ == '__main__':
    main()
