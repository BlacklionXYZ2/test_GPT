import random

maxProb = 100
startWord = 'the'

def load(file):
    f = open(f'EPQ_projects//weighting algorithms//separated algorithms//{file}', 'r')
    f = f.read()
    return eval(f)

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
w = load('weights.txt')
run(word = startWord, weights = w)
print(out)