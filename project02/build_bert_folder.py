import numpy as np

def divide_test_train(tweets):
	index = int(0.75 * len(tweets))
	return tweets[:index], tweets[index:]

def tweets_txt(file_name):
	tweets_txt = []
	f = open(file_name, "r")
	for l in f.readlines():
		tweets_txt.append(l.strip())
	f.close()
	return np.array(list(set(tweets_txt)))

def main():
	# https://towardsdatascience.com/bert-text-classification-in-3-lines-of-code-using-keras-264db7e7a358

	tweets_pos = tweets_txt("Datasets/twitter-datasets/train_pos.txt")
	tweets_neg = tweets_txt("Datasets/twitter-datasets/train_neg.txt")

	np.random.seed(42)

	tweets_pos_train, tweets_pos_test = divide_test_train(tweets_pos)
	tweets_neg_train, tweets_neg_test = divide_test_train(tweets_neg)

	i = 0
	for t in tweets_pos_train:
		f= open("BERT_folder_small/train/pos/%d.txt" %i,"w+")
		f.write(t)
		f.close()
		i+=1
	print("DONE pos train")
    
	i = 0
	for t in tweets_pos_test:
		f= open("BERT_folder_small/test/pos/%d.txt" %i,"w+")
		f.write(t)
		f.close()
		i+=1

	print("DONE pos test")

	i = 0
	for t in tweets_neg_train:
		f= open("BERT_folder_small/train/neg/%d.txt" %i,"w+")
		f.write(t)
		f.close()
		i+=1

	print("DONE neg train")

	i = 0
	for t in tweets_neg_test:
		f= open("BERT_folder_small/test/neg/%d.txt" %i,"w+")
		f.write(t)
		f.close()
		i+=1

	print("DONE neg test")

	
if __name__ == "__main__":
    # execute only if run as a script
    main()
