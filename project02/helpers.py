import numpy as np 

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
    tmp = np.hstack((X, y.reshape((len(y),1))))
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
