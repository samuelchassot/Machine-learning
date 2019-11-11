#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat Datasets/twitter-datasets/train_pos.txt Datasets/twitter-datasets/train_neg.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab.txt

# sed "s/ /\n/g" replaces every occurence (g flag : "/g") of the space character ("/ ") with line return character ("/\n")
# grep -v "^\s*$" filters the input and outputs all lines that do not contain the given regular expression
# sort sorts the file
# uniq -c write a copy of each unique input line with the number of occurences before
