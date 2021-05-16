from LSTM import headline, tweets


def main():
    pass


if __name__ == '__main__':

    sentence = ["God, you are the best boss EVER! Have I ever told you how much I love this job? I  wish I could live here! Somebody get me a tent, I never wanna leave!"]
    # print(headline(sentence))
    print(tweets(sentence))
    # sentence = tokenizer.texts_to_sequences(sentence)
    # sentence = pad_sequences(sentence, maxlen=29, dtype='int32', value=0)

    # loaded_model = keras.models.load_model("model")

    # sentiment = loaded_model.predict(sentence, batch_size=1, verbose=2)[0]

    # if np.argmax(sentiment) == 0:
    #     print("Non-sarcastic")
    # elif np.argmax(sentiment) == 1:
    #     print("Sarcasm")
