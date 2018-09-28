#!/usr/bin/env python3
from keras.layers import *
from keras.models import *

from torchtext import data, datasets

from kltt import WrapIterator


# Model parameters
batch_size = 64
epochs = 3
seq_length = 100
top_words = 10000
embedding_dim = 300
lstm_units = 128

print('Loading and preparing data')

text_field = data.Field(fix_length=seq_length)
# `unk_token` should be set to None if the label data is expected to be categorical and to have no inconsistencies
# like classes in the test set that didn't appear in the training set.
label_field = data.Field(sequential=False, unk_token=None)

train_set, test_set = datasets.IMDB.splits(text_field, label_field)
train_it, test_it = data.BucketIterator.splits([train_set, test_set], [batch_size] * 2, repeat=True)

text_field.build_vocab(train_set, max_size=top_words)
label_field.build_vocab(train_set)

print('Vocabulary contains {} words'.format(len(text_field.vocab)))

train_data, test_data = WrapIterator.wraps([train_it,
                                            test_it], ['text'], ['label'], permute={'text': (1, 0)})

print('Building model')

model = Sequential()
model.add(Embedding(top_words, embedding_dim))
model.add(LSTM(lstm_units))
# Actually the labels are binary but we will treat it as a categorical problem for seek of generality
model.add(Dense(len(label_field.vocab), activation='softmax'))

model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

print('Training model')

model.fit_generator(iter(train_data), steps_per_epoch=len(train_data), epochs=epochs)

# Evaluate
loss, acc = model.evaluate_generator(iter(test_data), steps=len(test_data))

print('Test loss: {:.4f}'.format(float(loss)))
print('Test accuracy: {:.2f}%'.format(float(acc) * 100.0))