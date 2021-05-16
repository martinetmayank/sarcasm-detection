from LSTM import lstm_headline, lstm_tweets
from Bayes import bayes

import os


def get_result(sentence):
    l1 = lstm_tweets(sentence)
    l2 = lstm_headline(sentence)
    b = bayes(sentence[0])

    return {
        'twitter LSTM': l1,
        'headline LSTM': l2,
        'Naive bayes': b
    }


def clear():
    os.system('cls')
