import random

#////////////////////////////////
#outdated and not working
#///////////////////////////////

class weights:
    def find(data):
        f = open(f'EPQ_projects//{data}', 'r')
        f = f.read()
        f = f.split()
        
        t = set(f)
        words = {}

        for x in t:
            words[x] = [0]

        for x in f:
            for y in words:
                if y == x:
                    words[x][0] += 1
                    

        for w1, w2 in zip(f[:-1], f[1:]):
            words[w1].append([w2, 1])
            

        print(f)
        print(words)

weights.find('test.txt')