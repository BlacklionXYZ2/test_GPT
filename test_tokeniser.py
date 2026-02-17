import re

file = 'training.txt'

with open(f'EPQ_projects//{file}', 'r', encoding = 'utf-8') as f:
    file = f.read()
    t = re.split(r'([,.:;?_!"()\']|--|\s)', file)
    t = [item.strip() for item in t if item.strip()]

words = sorted(set(t))
words.extend(['<|endoftext|>', '<|unknown|>'])
size = len(words)

vocab = {token: integer for integer, token in enumerate(words)}

class tokeniserClass:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        processed = re.split(r'([,.:;?_!"()\']|--|\s)', text) 
        processed = [item.strip() for item in processed if item.strip()]
        processed = [item if item in self.str_to_int else '<|unknown|>' for item in processed]
        ids = [self.str_to_int[x] for x in processed]
        return ids

    def decode(self, ids):
        text = ' '.join([self.int_to_str[x] for x in ids])

        text = re.sub(r'\s+([,.?"()\'])', r'\1', text)
        return text
    
tokeniser = tokeniserClass(vocab)
text = 'the cat sat on the mat'
ids = (tokeniser.encode(text = text))
out = tokeniser.decode(ids)
print(out)