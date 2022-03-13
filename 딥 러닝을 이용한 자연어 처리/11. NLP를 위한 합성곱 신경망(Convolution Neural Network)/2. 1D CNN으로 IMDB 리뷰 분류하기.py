 #1. 데이터에 대한 전처리
from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size=10000
(X_train, y_train), (X_test, y_test)=datasets.imdb.load_data(num_words=vocab_size)
print("상위 5개의 X_train: ")
print(X_train[:5])#전처리 이미 됨
print('상위 5개의 y_train:')
print(y_train[:5])

max_len=200
X_train=pad_sequences(X_train, maxlen=max_len)
X_test=pad_sequences(X_test, maxlen=max_len)

print("\nX_train의 크기(shape): ", X_train.shape)
print('\nX_test의 크기(shape): ', X_test.shape)

 #2. 1D CNN으로 IMDB 리뷰 분류하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

embedding_dim=256
dropout_ratio=0.3
num_filters=256
kernel_size=3
hidden_units=128

model=Sequential()
model.add(Embedding(vocab_size, embedding_dim))#vocab_size->embedding_dim
model.add(Dropout(dropout_ratio))
model.add(Conv1D(num_filters, kernel_size, padding='valid', activation='relu'))
model.add(GlobalMaxPooling1D())#scalar인데 num_filters가 256인
model.add(Dense(hidden_units, activation='relu'))#출력이 hidden_units(128)
model.add(Dropout(dropout_ratio))
model.add(Dense(1, activation='sigmoid'))#그리고 그 128이 1개로 by sigmoid

es=EarlyStopping(monitor='val_liss', mode='min', verbose=1, patience=3)
mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history=model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[es,mc])

loaded_model=load_model('best_model.h5')
print("\n테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
