from LSTM import lstm_headline, lstm_tweets
from Bayes import bayes


def main():
    pass


if __name__ == '__main__':

    sentence = [
        "mom starting to fear son's web series closest thing she will have to grandchild"
    ]

    l1 = lstm_tweets(sentence)
    l2 = lstm_headline(sentence)
    b = bayes(sentence[0])
