import nltk
from nltk.corpus import movie_reviews
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
import codecs
from nltk.tokenize import sent_tokenize, word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return max(set(votes), key=votes.count)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(max(set(votes), key=votes.count))
        conf = choice_votes / len(votes)
        return conf


#
# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]
#
# random.shuffle(documents)
#
# all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
# word_features = all_words.most_common(2000)

# short_pos = str(open("positive.txt","r").read())
# short_neg = str(open("negative.txt","r").read())

print("Reading positive...")
with codecs.open("positive.txt", "r", "latin-1") as inputfile:
    short_pos=inputfile.read()

print("Reading negative...")
with codecs.open("negative.txt", "r", "latin-1") as inputfile:
    short_neg=inputfile.read()


documents = []



for line in short_pos:
    documents.append( str((line, "pos")) )



for line in short_neg:
    documents.append( str((line, "neg") ))


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [(find_features(d), c) for (d, c) in documents]
random.shuffle(featuresets)

# def document_features(document):
#     document_words = set(document)
#     features = {}
#
#     for (word, val) in word_features:
#         features['contains(%s)' % word] = (word in document_words)
#
#     return features


#featuresets = [(document_features(d), c) for (d, c) in documents]
training_set = featuresets[:10000]
testing_set =  featuresets[10000:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print(nltk.classify.accuracy(classifier, testing_set))

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:" + str(nltk.classify.accuracy(MNB_classifier, testing_set)))




BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)



LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)



voted_classifier=VoteClassifier(classifier,NuSVC_classifier,LinearSVC_classifier,SGDClassifier_classifier,LogisticRegression_classifier,BernoulliNB_classifier,MNB_classifier)
print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


#print("classification",voted_classifier.classify(testing_set[0][0]))
# save_classifier=open("naivebayes.pickle","wb") #wb=write in bytes
# pickle.dump(classifier,save_classifier)
# save_classifier.close()
