import ktrain
from ktrain import text
import numpy as np

def divide_test_train(tweets):
    """Divide an array of tweet into two parts, one containing 75% and the other one, 25%"""
    index = int(0.75 * len(tweets))
    return tweets[:index], tweets[index:]

def tweets_txt(file_name):
    """Parse txt files to obtain an array of tweets and remove duplicates"""
    tweets_txt = []
    f = open(file_name, "r")
    for l in f.readlines():
        tweets_txt.append(l.strip())
    f.close()
    return np.array(list(set(tweets_txt)))

def tweets_txt_test(file_name):
    """Parse a txt file and return an array of tweets without removing duplicates"""
    tweets_txt = []
    f = open(file_name, "r")
    for l in f.readlines():
        tweets_txt.append(l.strip())
    f.close()
    return np.array(tweets_txt)


print("PREPARE FOLDER FOR BERT")

tweets_pos = tweets_txt("Datasets/twitter-datasets/train_pos.txt")
tweets_neg = tweets_txt("Datasets/twitter-datasets/train_neg.txt")

tweets_pos_train, tweets_pos_test = divide_test_train(tweets_pos)
tweets_neg_train, tweets_neg_test = divide_test_train(tweets_neg)

i = 0
for t in tweets_pos_train:
    f= open("BERT_folder/train/pos/%d.txt" %i,"w+")
    f.write(t)
    f.close()
    i+=1
print("DONE POS TRAIN")

i = 0
for t in tweets_pos_test:
    f= open("BERT_folder/test/pos/%d.txt" %i,"w+")
    f.write(t)
    f.close()
    i+=1

print("DONE POS TEST")

i = 0
for t in tweets_neg_train:
    f= open("BERT_folder/train/neg/%d.txt" %i,"w+")
    f.write(t)
    f.close()
    i+=1

print("DONE NEG TRAIN")

i = 0
for t in tweets_neg_test:
    f= open("BERT_folder/test/neg/%d.txt" %i,"w+")
    f.write(t)
    f.close()
    i+=1

print("DONE NEG TEST")

print("DONE PREPARING BERT FOLDER")

print("START TRAINING")

(x_train_small, y_train_small), (x_test_small, y_test_small), preproc_small = text.texts_from_folder("BERT_folder", 
	maxlen=199, 
    preprocess_mode='bert',
    train_test_names=['train', 'test'],
    classes=['pos', 'neg'])

model_small = text.text_classifier('bert', (x_train_small, y_train_small), preproc=preproc_small)
learner_small = ktrain.get_learner(model_small,train_data=(x_train_small, y_train_small), 
	val_data=(x_test_small, y_test_small), batch_size=10)

learner_small.fit_onecycle(2e-5, 1)


print("DONE WITH TRAINING")

print("START TO PREDICT")

predictor = ktrain.get_predictor(learner_small.model, preproc_small)

tweets_test = tweets_txt_test("Datasets/twitter-datasets/test_data.txt")

result = predictor.predict(tweets_test)

print("DONE TO PREDICT")

print("DO THE SUBMISSION FILE")

# make csv
with open("submission.csv", "w") as f:
    f.write("Id,Prediction\n")
    id = 1
    for i in result:
        if i == "neg":
            i = -1
        if i == "pos":
            i = 1
        l = str(id) + "," + str(i) + "\n"
        f.write(l)
        id = id + 1






