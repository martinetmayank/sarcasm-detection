import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')


def headline(sentence):

    sentence = tokenizer.texts_to_sequences(sentence)
    sentence = pad_sequences(sentence, maxlen=29, dtype='int32', value=0)

    loaded_model = keras.models.load_model("models/lstm-headline")

    sentiment = loaded_model.predict(sentence, batch_size=1, verbose=2)[0]

    if np.argmax(sentiment) == 0:
        print("Non-sarcastic")
        return 'Non-Sarcastic'
    elif np.argmax(sentiment) == 1:
        print("Sarcasm")
        return 'Sarcastic'


def tweets(sentence):
    # max_features = 2000
    # tokenizer = Tokenizer(num_words=max_features, split=' ')

    sentence = tokenizer.texts_to_sequences(sentence)
    sentence = pad_sequences(sentence, maxlen=29, dtype='int32', value=0)

    loaded_model = keras.models.load_model("models/lstm-tweets")

    sentiment = loaded_model.predict(sentence, batch_size=1, verbose=2)[0]

    if np.argmax(sentiment) == 0:
        print("Non-sarcastic")
        return 'Non-Sarcastic'
    elif np.argmax(sentiment) == 1:
        print("Sarcasm")
        return 'Sarcastic'
