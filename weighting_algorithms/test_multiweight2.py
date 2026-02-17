import random
from collections import defaultdict


maxProb = 100
file = 'test.txt'
startWord = 'the'


def findWeights(data):
    global maxProb
    f = open(f'EPQ_projects//weighting algorithms//{data}')
    f = f.read()
    f = f.split()
    weights = defaultdict(dict)

    for w1, w2 in zip(f[:-1], f[1:]):
        try:
            weights[w1][w2][0] += maxProb
        except KeyError:
            weights[w1][w2] = [maxProb, None]

    for x in weights:
        total = 0
        for y in weights[x]:
            total += weights[x][y][0]
        weights[x].update({'blocks': [maxProb / total, maxProb]})

    for x in weights:
        start = maxProb
        start2 = None
        idx = 0
        for y in weights[x]:
            if idx == 1:
                weights[x][y] = [start2[1], start2[1] + round((weights[x][y][0] * weights[x]['blocks'][0]))]
            if idx == 0:
                weights[x][y] = [maxProb - start, round((maxProb - start) + (weights[x][y][0] * weights[x]['blocks'][0]))]
                idx = 1
            start2 = weights[x][y]
        weights[x].pop('blocks')

    for x in weights:
        for key, value in enumerate(weights[x]):
            if key == len(weights[x]) - 1:
                weights[x][value][1] = maxProb
    return weights

def predict(word, weights):
    global out
    nextWord = random.randint(0, maxProb)
    for x in weights[word]:
        if nextWord in list(range(weights[word][x][0], weights[word][x][1] + 1)):
            out += x + ' '
            return x

def run(word, weights):
    global out
    out = word + ' '
    while word in weights:
        word = predict(word, weights)

out = None
w = findWeights(file) 
run(word = startWord, weights = w)
print(out)