#import tweepy
import pickle
import nltk
import re
from googletrans import Translator
translator = Translator()

nltk.download('punkt')

classifier = pickle.load(open('data_classification/MNB.pickle', 'rb'))
word_features = pickle.load(open('data_classification/word_features.pickle', 'rb'))

def document_features(document):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

def tweet_clean(t):
		t = t.replace("#", "")
		t = t.replace("@", "")
		t = re.sub(r"[^\w\s]","",t)
		t = re.sub(" \d+", " ", t)
		return t

def predict_topic(s):
	s = tweet_clean(s)
	token = nltk.word_tokenize(s.lower())
	return classifier.classify(document_features(token))

def predict_tweet_topic(s):
    tt = translator.translate(tweet_clean(s))
    return tt.src, tt.text, predict_topic(tt.text)
    
#a = predict_topic('how was real madrid game last night?')
#print(translator.translate("salut comment allez-vous"))
#print(translator.translate("Hallo wie geht's dir"))

#print(translator.translate('amazing to see how irans deal goes').text)
#print(translator.translate('amazing to see how irans deal goes').src)

