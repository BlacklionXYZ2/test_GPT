import tiktoken

tokeniser = tiktoken.get_encoding('gpt2')
with open('EPQ_projects//training.txt', 'r', encoding = 'utf-8') as f:
    rawText = f.read()

encodedText = tokeniser.encode(rawText)
sample = encodedText[50:]
contextSize = 4
x = sample[:contextSize]
y = sample[1:contextSize + 1]

for x in range(1, contextSize + 1):
    context = sample[:x]
    target = sample[x]