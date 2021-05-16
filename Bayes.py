from sklearn.externals import joblib
import re
from nltk.stem.porter import PorterStemmer


bnb = joblib.load('models/bayes/bnb.pkl')

tf = joblib.load('models/bayes/tf.pkl')


def bayes(headline):
    review = re.sub('[^a-zA-Z]', ' ', headline)
    review = review.lower()
    review = review.split()

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    review = [review]

    test = tf.transform(review).toarray()

    labels_pred = bnb.predict(test)
    print(type(labels_pred))
    if labels_pred[0] == "1":
        return "Sarcasm"
    else:
        return "no Sarcasm"
