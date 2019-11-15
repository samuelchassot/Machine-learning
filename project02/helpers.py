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

def tweet_means(file_name, word_embeddings, words_list, embedding_size):
    tweets_txt = []
    f = open(file_name, "r")
    for l in f.readlines():
        tweets_txt.append(l.strip())
    tweets_pos_txt = np.array(tweets_txt)
    f.close()

    tweets_vec = []
    for tw in tweets_txt:
        words_in_tweet = tw.split(" ")
        acc = np.zeros(embedding_size)
        for w in words_in_tweet:
            vec = word_embeddings[np.argmax(words_list==w)]
            acc += vec
        acc = acc/len(words_in_tweet)
        tweets_vec.append(acc)
    tweets_vec = np.array(tweets_vec)
    return tweets_vec


