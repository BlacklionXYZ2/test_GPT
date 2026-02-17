import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn


#1. Tokenisation - how does text become numbers
text = 'the cat sat on the mat'
vocab = {word : i for i, word in enumerate(set(text.split()))}
tokens = [vocab[word] for word in text.split()]

print('Words: ', vocab)
print('Tokens: ', tokens)

# Tokenisation works by taking a list of words and converting them into a set, which can only contain one of each
# entry. This is then enumerated and the result is effectively a filtered list of words with numbers assigned to 
# them. The numbers are then taken as the output, which are the tokens that the model will use later


#2. Vectors - how do arbitrary values become meaningful vectors
vocabSize = len(vocab)
def vectorise(index, size = vocabSize):
    vector = np.zeros(size)
    vector[index] = 1
    return vector

vectors = []
for x in tokens:
    vectors.append(vectorise(x))
print(vectors)

# The function creates an array with a length equal to the number of different words there are and sets the value 
# with the same index as the specified word to 1


#3. Prediction - how can numbers predict numbers
bigram_counts = defaultdict(lambda: defaultdict(int))
for w1, w2 in zip(text.split()[:-1], text.split()[1:]):
    bigram_counts[w1][w2] += 1

bigram_probs = defaultdict(dict)

for w1, next_words in bigram_counts.items():
    total = sum(next_words.values())
    for w2, count in next_words.values(): # fix w2, count later
        bigram_probs[w1][w2] = count / total

print('Counts', bigram_counts)
print('Probs', bigram_probs)

#figured out word prediction in another file using fixed weights
#skip and work on #4