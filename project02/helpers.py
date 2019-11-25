import numpy as np 
from pattern.en import *

def words_list(file_name):
    words_list = []
    f = open(file_name, "r")
    for l in f.readlines():
        l = l.strip()
        words_list.append(l)
    words_list = np.array(words_list)
    f.close()
    return words_list

def tweets_txt(file_name):
    tweets_txt = []
    f = open(file_name, "r")
    for l in f.readlines():
        tweets_txt.append(l.strip())
    f.close()
    return np.array(tweets_txt)

def tweet_means(tweets_txt, word_embeddings, words_list, embedding_size, clean = False, common = []):
    tweets_vec = []
    for tw in tweets_txt:
        words_in_tweet = tw.split(" ")
        if(clean):
            words_in_tweet = remove_words(words_in_tweet, common)
            words_in_tweet = remove_exclamation(words_in_tweet)
        acc = np.zeros(embedding_size)
        for w in words_in_tweet:
            vec = word_embeddings[np.argmax(words_list==w)]
            acc += vec
        acc = acc/len(words_in_tweet)
        tweets_vec.append(acc)
    tweets_vec = np.array(tweets_vec)
    return tweets_vec

def remove_duplicated_tweets_txt(tweets):
    return np.unique(tweets, axis = 0)

def remove_duplicated_tweets(X, y):
    np.hstack((X, y.reshape((len(y),1))))
    tmp = np.unique(tmp, axis=0)

    new_X = tmp[:,:-1]
    new_y= tmp[:,-1]
    return new_X, new_y

def remove_words(words, tweet):
    """ Remove the words that are in the list <words> from the tweets """
    filtered_tweet = [w for w in tweet if w not in words]   
    return filtered_tweets

def remove_exclamation(tweet):
    """ Remove the "!!!" that may be at the beginning of tweets """
    if(tweet[0:3] == ['!', '!', '!']):
        return tweet[3:]
    return tweet

def transform_negation(tweet):
    """Transform a negated verb into an infinitive form + not
        ex: don't -> do not
    """
    new_tweet = []
    for w in tweet:
        #We check if the verb is negated and if the pattern library knows its infinitive form
        if("n't" in w and conjugate(w) != w):
            new_tweet.append(conjugate(w))
            new_tweet.append("not")
        else:
            new_tweet.append(w)
    
    return w

def transform_spelling(tweet, spelling_dict):
    """ Replace the words of a tweet by another spelling if they are in <spelling_dict> """
    new_tweet = [spelling_dict.get(w, w) for w in tweet]
    return new_tweet
