from matplotlib import pyplot
import tensorflow
import seaborn
import pandas
import numpy
import sys
from multiprocessing import Pool, get_context
from concurrent.futures import ProcessPoolExecutor
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

from tensorflow.keras import Model
from keras.models import load_model

from tensorflow.keras import layers
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Conv2D, SeparableConv2D, LocallyConnected2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler

print("Tensorflow version", tensorflow.__version__)
if tensorflow.test.gpu_device_name() != '':
    print('Connected to GPU ' + tensorflow.test.gpu_device_name())
else:
    print('Not connected to GPU')


class DCNN_classifier:
    def __init__(self, model_name='Baseline_model', batch_size=32, epochs=15, kfold=5, conv_blocks=[64],
                 kernel_size=(3, 3),
                 kernel_regularization='L1', kernel_initialization='he_normal', optimizer=SGD, learning_rate=1e-3,
                 classification_style='soft_max'):
        self.model_name = model_name
        self.model = None

        self.batch_size = batch_size
        self.epochs = epochs
        self.kfold = kfold

        self.conv_blocks = conv_blocks
        self.kernel_size = kernel_size
        self.kernel_regularization = kernel_regularization
        self.kernel_initialization = kernel_initialization

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.classification_style = classification_style

        self.histories = []
        self.show_model_char = True

    # load train and test dataset
    def load_dataset(self):
        # load dataset
        (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
        # reshape dataset to have a single channel
        train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
        test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
        # one hot encode target values
        train_y_1hot = tensorflow.keras.utils.to_categorical(train_y)
        test_y_1hot = tensorflow.keras.utils.to_categorical(test_y)

        return [train_x, train_y_1hot, train_y, test_x, test_y_1hot, test_y]

    # data augmentation
    def data_augmentation(self, data):
        augmented_data = ImageDataGenerator(fill_mode='constant',  # 'nearest'
                                            horizontal_flip=True,  #vertical_flip=True,
                                            width_shift_range=0.2, height_shift_range=0.2,
                                            #zoom_range=0.6, rotation_range=15,
                                            dtype=float)
        augmented_data.fit(data)
        return augmented_data

    # define model
    def define_model(self, input_shape, DEPTH=[128], KERNEL_SIZE=(3, 3), KERNEL_REGULARIZER='l1',
                     KERNEL_INITIALIZER='he_normal'):
        input = tensorflow.keras.Input(shape=input_shape)
        input = Rescaling(1. / 255)(input)

        # Entry block
        x = SeparableConv2D(filters=DEPTH[0] * 2, kernel_size=KERNEL_SIZE,
                            kernel_regularizer=KERNEL_REGULARIZER, kernel_initializer=KERNEL_INITIALIZER,
                            padding='same')(input)
        x = Activation('tanh')(x)
        x = BatchNormalization()(x)
        prev_act = x

        # Convolution block
        for block, num_filters in enumerate(DEPTH):
            x = SeparableConv2D(filters=num_filters, kernel_size=KERNEL_SIZE,
                                kernel_regularizer=KERNEL_REGULARIZER, kernel_initializer=KERNEL_INITIALIZER,
                                padding='same')(x)
            x = Activation('tanh')(x)
            x = BatchNormalization()(x)

            x = SeparableConv2D(filters=num_filters, kernel_size=KERNEL_SIZE,
                                kernel_regularizer=KERNEL_REGULARIZER, kernel_initializer=KERNEL_INITIALIZER,
                                padding='same')(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = BatchNormalization()(x)

            x = SeparableConv2D(filters=num_filters, kernel_size=KERNEL_SIZE,
                                kernel_regularizer=KERNEL_REGULARIZER, kernel_initializer=KERNEL_INITIALIZER,
                                padding='same')(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = BatchNormalization()(x)

            # Residual connection
            residual = SeparableConv2D(filters=num_filters, kernel_size=(1, 1),
                                       kernel_regularizer=KERNEL_REGULARIZER, kernel_initializer=KERNEL_INITIALIZER,
                                       padding='same')(prev_act)
            x = BatchNormalization()(x)

            x = layers.add([x, residual])  # Add back residual
            x = Dropout(0.2)(x)
            prev_act = x # Set aside next residual

        # Exit block
        x = SeparableConv2D(filters=DEPTH[-1] * 2, kernel_size=KERNEL_SIZE,
                            kernel_regularizer=KERNEL_REGULARIZER, kernel_initializer=KERNEL_INITIALIZER,
                            padding='same')(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Activation('tanh')(x)
        output = BatchNormalization()(x)

        model = Model(inputs=input, outputs=output, name=f'{self.model_name}_Basis')
        return model

    # compile model
    def compile_model(self, model, num_classes, show_model=False, OPTIMIZER=SGD, LEARNING_RATE=1e-3, FLAG=''):
        output = Dropout(0.5)(model.output)
        output = Flatten()(output)

        # Classification layer (support vector machine)
        if FLAG == 'svm':
            output = Dense(units=num_classes, kernel_regularizer='l2')(output)
            output = Activation('linear')(output)

            loss = 'categorical_hinge'
            metrics = ['accuracy']

        # Classification layer (softmax)
        else:
            output = Dense(units=num_classes)(output)
            output = Activation('softmax')(output)

            loss = 'categorical_crossentropy'
            metrics = ['accuracy', 'Precision', 'Recall', 'AUC']

        # compile model
        model = Model(inputs=model.inputs, outputs=output, name=f'{self.model_name}_InclTop')
        model.compile(optimizer=OPTIMIZER(learning_rate=LEARNING_RATE, decay=LEARNING_RATE),
                      loss=loss, metrics=metrics)  # 'categorical_hinge'
        # show model
        if show_model:
            plot_model(model, f'{self.model_name}_Graph.png', show_shapes=True, dpi=20 * len(model.layers), expand_nested=True)
            model.summary()
        return model

    # adjust the learning rate dynamically across epochs
    def dynamical_decay(self, epoch, lr, factor=0.5):
        if epoch % 5 == 0 and epoch != 0:
            print(f'learning_rate has changed to {numpy.round(lr * factor, 7)}')
            return lr * factor
        return lr

    def evaluate_model(self, model, fold, EPOCHS=15, BATCH_SIZE=32):
        # callbacks
        #check_points = ModelCheckpoint([f'{self.model_name}_checkpoint.h5'], save_best_only=True,
        #                               monitor='val_loss', verbose=0)
        dynamical_learning_rate = LearningRateScheduler(self.dynamical_decay)
        early_stopping = EarlyStopping(monitor='val_loss', patience=6)

        # data augmentation
        trainGenerator = self.data_augmentation(fold[0]).flow(x=fold[0], y=fold[1], batch_size=BATCH_SIZE)
        valGenerator = self.data_augmentation(fold[2]).flow(x=fold[2], y=fold[3], batch_size=BATCH_SIZE)
        # fit model
        history = model.fit(x=fold[0], y=fold[1], validation_data=[fold[2], fold[3]], # x=trainGenerator, validation_data=valGenerator,
                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, shuffle='TRUE',
                            callbacks=[early_stopping, dynamical_learning_rate])
        # evaluate model
        if self.classification_style == 'svm':
            _, acc = model.evaluate(x=fold[2], y=fold[3], verbose=0)
            print(f' \n Model evaluation: Accuracy = {numpy.round(acc * 100.0, 3)}% \n')
        else:
            _, acc, precs, rec, _ = model.evaluate(x=fold[2], y=fold[3], verbose=0)
            print(f' \n Model evaluation: Accuracy = {numpy.round(acc * 100.0, 3)}%, '
                  f'Precision = {numpy.round(precs * 100.0, 3)}%, Recall = {numpy.round(rec * 100.0, 3)}% \n')

        return history

    # plot diagnostic learning curves
    @staticmethod
    def summarize_diagnostics(self):
        pyplot.figure(figsize=(13, 9))
        pyplot.suptitle(f'Performance of model: {self.model_name}', fontsize=20)

        for i, history in enumerate(self.histories):
            # plot loss
            if self.classification_style == 'svm': pyplot.subplot(1, 2, 1)
            else: pyplot.subplot(2, 2, 1)
            pyplot.title('Cross Entropy Loss', fontsize=14)
            pyplot.plot(history.history['loss'], color='blue', label='train')
            pyplot.plot(history.history['val_loss'], color='orange', label='test')
            pyplot.ylabel('Loss', fontsize=10)
            pyplot.xlabel('Epoch', fontsize=10)
            # plot accuracy
            if self.classification_style == 'svm': pyplot.subplot(1, 2, 2)
            else: pyplot.subplot(2, 2, 2)
            pyplot.title('Classification Accuracy', fontsize=14)
            pyplot.plot(history.history['accuracy'], color='blue', label='train')
            pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
            pyplot.ylabel('Accuracy', fontsize=10)
            pyplot.xlabel('Epoch', fontsize=10)

            if self.classification_style != 'svm':
                # plot precision
                pyplot.subplot(2, 2, 3)
                pyplot.title('Classification Precision', fontsize=14)
                pyplot.plot(history.history['precision'], color='blue', label='train')
                pyplot.plot(history.history['val_precision'], color='orange', label='test')
                pyplot.ylabel('Precision', fontsize=10)
                pyplot.xlabel('Epoch', fontsize=10)
                # plot recall
                pyplot.subplot(2, 2, 4)
                pyplot.title('Classification Recall', fontsize=14)
                pyplot.plot(history.history['recall'], color='blue', label='train')
                pyplot.plot(history.history['val_recall'], color='orange', label='test')
                pyplot.ylabel('Recall', fontsize=10)
                pyplot.xlabel('Epoch', fontsize=10)

        # save plots to a file
        pyplot.savefig(f'{self.model_name}_PerfMetric.png')

    # function to draw confusion matrix
    @staticmethod
    def draw_confusion_matrix(self, true, preds):
        conf_matx = confusion_matrix(true, preds)
        ax = pyplot.subplot()
        seaborn.heatmap(conf_matx, annot=True, annot_kws={"size": 12},
                        fmt='g', ax=ax, cmap='Greens', cbar=False, )
        pyplot.suptitle(f'Confusion Matrix of model: {self.model_name}', fontsize=16)

        # save confusion matrix to a file
        pyplot.savefig(f'{self.model_name}_ConfMat.png')


    # summarize performance on the test set
    @staticmethod
    def classification_report(self, predictions, testLabels):
        report = classification_report(predictions, testLabels, output_dict=True)

        # save classification report to a file
        pandas.DataFrame(report).transpose().to_csv(f'{self.model_name}_Report.csv', index=True)

    # run the test harness for evaluating a model
    def run_DCNN(self):

        # load dataset
        trainX, trainY, _, testX, testY, testLabels = self.load_dataset()  # trainX, trainY_1hot, trainY, testX, testY_1hot, testY

        # cross validation
        cur_fold = 1
        # prepare cross validation
        kfold = KFold(n_splits=self.kfold, shuffle=True, random_state=1)  # 5 folds as default
        # enumerate splits
        for train_ix, val_ix in kfold.split(trainX):
            print(f'Running corss-validation: {cur_fold}/{kfold.n_splits}.')
            # select rows for train and test
            fold = (trainX[train_ix], trainY[train_ix], trainX[val_ix], trainY[val_ix])

            # define and compile the model
            print(f'[INFO] compiling model...')
            base_model = self.define_model(input_shape=fold[0].shape[1:], DEPTH=self.conv_blocks, KERNEL_SIZE=self.kernel_size,
                                           KERNEL_REGULARIZER=self.kernel_regularization, KERNEL_INITIALIZER=self.kernel_initialization)
            self.model = self.compile_model(model=base_model, num_classes=len(fold[1][0]), show_model=self.show_model_char,
                                            OPTIMIZER=self.optimizer, LEARNING_RATE=self.learning_rate, FLAG=self.classification_style)
            # fit and evaluate the model
            print(f'[INFO] training model...')
            history = self.evaluate_model(model=self.model, fold=fold, EPOCHS=self.epochs, BATCH_SIZE=self.batch_size)

            self.histories.append(history)
            self.show_model_char = False
            cur_fold += 1

        # learning curves
        self.summarize_diagnostics(self)

        print(f'[INFO] testing model on test set...\n')
        # predict on test set
        predictions = numpy.argmax(self.model.predict(testX), axis=1)
        # draw a confusion matrix
        self.draw_confusion_matrix(self, predictions, testLabels)  # test_x, test_y
        # classification report
        self.classification_report(self, predictions, testLabels)

# entry point, run the DCN
def main():
    model_types = [DCNN_classifier(model_name='BaselineModel1', conv_blocks=[128, 256], kfold=2, optimizer=RMSprop, epochs=8)]
    # model_types = [DCNN_classifier(model_name='BaselineModel', conv_blocks=[128], kfold=2),
    #                DCNN_classifier(model_name='RMSpropModel', conv_blocks=[128], kfold=2, optimizer=RMSprop),
    #                DCNN_classifier(model_name='4x4KernelModel', conv_blocks=[128], kfold=2, kernel_size=(4, 4)),
    #                DCNN_classifier(model_name='DeepModel', kfold=2, conv_blocks=[128, 256]),
    #                DCNN_classifier(model_name='SVMModel, kfold=2', conv_blocks=[128], kfold=2, classification_style='svm')]
    best_score = float('-inf')

    for idx, model_type in enumerate(model_types):
        classifier = model_type
        classifier.run_DCNN()

        if classifier.histories[-1].history['accuracy'][-1] < best_score:
            best_score = classifier.histories[-1].history['accuracy'][-1]  # !!!!!!!!!!!!!!! NEED SIMONS CODE !!!!!!!!!!!!!!!!!!!!!!!!
            classifier.model.save(f'{classifier.model_name}.h5')
        del classifier.model

if __name__ == '__main__':
    main()

# TODO parallel processing
    # PUT under load data in main()
    # Histories = []
    # ShowModelInfo = True
    # # prepare cross validation
    # kfold = KFold(n_splits=4, shuffle=True, random_state=1)
    # from functools import partial
    # func = partial(cross_validate, trainX, trainY, Histories, ShowModelInfo)
    # with get_context('spawn').Pool(processes=4) as pool:
    #     splits = [(train_ix, val_ix) for train_ix, val_ix in kfold.split(trainX)]
    #     pool.imap_unordered(func, splits) #, ShowModelInfo
    # sys.stdout.flush()
    # # learning curves
    # #summarize_diagnostics(Histor)
    # versions: (1)tanh+glorot_normal vs leakyRelu+he_normal, (2)softmax vs svm,
    # (3)[64,128] vs [64, 128, 512], (4)SGD vs RMSprop, (5)filter_size(3,3) vs filter_size(2,2)

# TODO cross-validation function
    # def cross_validate(self, train_ix, val_ix, trainX, trainY, ShowModelInfo, Histories, BATCH_SIZE=64, EPOCHS=15):
    #     # select rows for train and test
    #     Fold = (trainX[train_ix], trainY[train_ix], trainX[val_ix], trainY[val_ix])
    #
    #     # define and compile the model
    #     print(f'[INFO] compiling model...')
    #     sys.stdout.flush()
    #     BaseModel = self.define_model(input_shape=Fold[0].shape[1:], DEPTH=[64, 128]) # input_shape, DEPTH=[128], KERNEL_SIZE=(3, 3), KERNEL_REGULARIZER='l1', KERNEL_INITIALIZER='he_normal'
    #     FinalModel = self.compile_model(model=BaseModel, num_classes=len(Fold[1][0]), show_model=ShowModelInfo) # model, num_classes, show_model=False, OPT=SGD, LEARNING_RATE=1e-3, flag='softmax'):
    #     # fit and evaluate the model
    #     print(f'[INFO] training model...')
    #     History = self.evaluate_model(model=FinalModel, fold=Fold,
    #                                        BATCH_SIZE=BATCH_SIZE, EPOCHS=EPOCHS)
    #     Histories.append(History)
    #     ShowModelInfo = False
    #     sys.stdout.flush()
    #     return Histories #, ShowModelInfo

# TODO transfer-learning
    # def transfer_learning(self):
    #     # transfer learning for the full data set
    #     Full = (trainX, trainY, testX, testY) # get the full data set
    #     # define transfer-learning model
    #     BaseModel=load_model('Baseline_DCN.h5') # already saved in checkpoints
    #     # mark loaded layers as not trainable
    #     BaseModel = BaseModel.pop(BaseModel.layers[-4])
    #     for layer in BaseModel.layers: # might change: not saved in the loop but at the end  after some improvement criteria over the other models
    #         layer.trainable = False
    #     x = SeparableConv2D(filters=128, kernel_size=(3, 3), # should be hyperparamters
    #                         kernel_regularizer='l1', kernel_initializer='he_initialization',# should be hyperparameters
    #                         padding='same')(BaseModel.outputs)
    #     x = AveragePooling2D(pool_size=(3, 3), padding='same')(x)
    #     x = LeakyReLU(alpha=0.3)(x)
    #     x = BatchNormalization()(x)
    #     output = Dense(units=256)(x)
    #     TransLrBaseModel = Model(inputs=BaseModel.inputs, outputs=output)
    #     # compile the transfer-learning model
    #     self.model = self.compile_model(model=TransLrBaseModel, num_classes=len(Fold[1][0]), show_model=ShowModelChar) #(model=BaseModel, num_classes=len(Fold[1][0]), show_model=self.show_model_char, OPTIMIZER=SGD, LEARNING_RATE=self.learning_rate, FLAG=self.classification_style)
    #     # dit and evaluate the transfer-learning model to the full data set
    #     Histories = self.evaluate_model(model=self.model, fold=Full, histories=self.histories,
    #                                BATCH_SIZE=self.batch_size, EPOCHS=5)
    #
    #     # predict on test set
    #     predictions = numpy.argmax(self.model.predict(testX), axis=1)
    #     # draw a confusion matrix
    #     self.draw_confusion_matrix(predictions, testLabels)  # test_x, test_y
    #     # classification report
    #     print(f' Classification Report on Test set: \n {classification_report(predictions, testLabels)}')
