# tokenizing and padding the data for RNN
import streamlit as st
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import layers, regularizers
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

max_features = 2000
reg_strength = 0.005
def create_model(batch_size=32, num_layers=200, dropout_rate=0.5, embed_vec_size=120, learning_rate=0.0001, optimizer='adam'):
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=embed_vec_size))
    model.add(layers.LSTM(num_layers, dropout=dropout_rate, return_sequences=True))
    model.add(BatchNormalization())
    model.add(layers.LSTM(num_layers, dropout=dropout_rate))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=50, activation='relu', kernel_regularizer=regularizers.l2(reg_strength)))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def lstm(data_clean_shortened,data_labels_shortened):
    tokenizer = Tokenizer(max_features, split=' ')
    tokenizer.fit_on_texts(data_clean_shortened)
    X_tokenized = tokenizer.texts_to_sequences(data_clean_shortened)
    X_padded = pad_sequences(X_tokenized)

    X_train_rnn, X_test_rnn, y_train_rnn, y_test_rnn = train_test_split(X_padded, data_labels_shortened, test_size=0.2,
                                                                        random_state=4)
    model = Sequential()

    # hyperparams

    embed_vec_size = 120  # hyperparam that controls size of embedding vector
    num_layers = 200  # also hyperparam
    batch_size = 32
    learning_rate = 0.0001
    dropout_rate = 0.5
    reg_strength = 0.005
    patience = 10

    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam']
    # batch_size = [16,32,64,128,256]

    model.add(Embedding(input_dim=max_features, output_dim=embed_vec_size))
    model.add(layers.LSTM(num_layers, dropout=dropout_rate, return_sequences=True))
    model.add(BatchNormalization())
    model.add(layers.LSTM(num_layers, dropout=dropout_rate))  # Additional LSTM layer
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=50, activation='relu', kernel_regularizer=regularizers.l2(reg_strength)))
    # model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation='sigmoid'))

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train_rnn, y_train_rnn, epochs=100, batch_size=batch_size, validation_data=(X_test_rnn, y_test_rnn),
                     callbacks=[early_stopping])  # , callbacks=[early_stopping]
    score = model.evaluate(X_test_rnn, y_test_rnn)  # 2-element vector containing loss, and accuracy

    str_= 'Test loss: ', score[0]
    st.text(str_)
    str_ = 'Test accuracy: ', score[1]
    st.text(str_)

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    stop_epoch = early_stopping.stopped_epoch - patience
    plt.axvline(stop_epoch, color='r')
    label = 'stopping point: \nepoch=' + str(stop_epoch)
    plt.text(stop_epoch + 0.2, .8, label, rotation=0)
    st.pyplot(plt.show())

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.axvline(stop_epoch, color='r')
    plt.text(stop_epoch + 0.2, .82, label, rotation=0)
    st.pyplot(plt.show())

    param_grid = {
        # 'batch_size': [32, 64, 128],
        'num_layers': [150, 200, 250],
        # 'dropout_rate': [0.3, 0.5, 0.7],
        # 'embed_vec_size': [120, 160, 200],
        'learning_rate': [0.001, 0.0001, 0.00001],
        'optimizer': ['adam', 'sgd', 'rmsprop']
    }

    model = KerasClassifier(model=create_model, loss="binary_crossentropy", dropout_rate=0.5, embed_vec_size=120,
                            learning_rate=0.0001, num_layers=200)

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=2)
    grid_result = grid.fit(X_train_rnn, y_train_rnn, callbacks=[early_stopping])

    best_params = grid_result.best_params_
    accuracy = grid_result.best_score_

    st.text("Best Params: ", best_params)
    st.text("Best Accuracy: ", accuracy)

    st.text("Grid Search CV best accuracy: ", grid_result.best_estimator_.score(X_train_rnn, y_train_rnn))


