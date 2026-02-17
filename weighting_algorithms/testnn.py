import random

dataset = {"the": [["cat", [1, 5]], ["mat", [6, 10]]],
           "cat": [["sat", [1, 10]]],
           "sat": [["on", [1, 10]]],
           "on": [["the", [1, 10]]]}

out = 'a'

def predict(word):
   global out
   nextword = random.randint(1, 10)
   for x in dataset[word]:
      if nextword in list(range(x[1][0], x[1][1] + 1)):
         out += x[0] + ' '
         break
   return x[0]

def run(word):
   global out
   out += word + ' '
   while word in dataset:
      word = predict(word)

start = 'the'
run(start)
print(out[1:])