from collections import defaultdict

def findWeights(file, maxProb):
    f = open(f'EPQ_projects//weighting algorithms//separated algorithms//{file}')
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

save = findWeights('test.txt', 100)

f = open('EPQ_projects//weighting algorithms//separated algorithms//weights.txt', 'w')

f.write(str(save)[28:len(str(save)) - 1])