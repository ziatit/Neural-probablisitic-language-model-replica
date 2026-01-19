import torch
from collections import Counter

def preapare_tokens(words):
  tokens = [w.lower() for w in words]
  freq = Counter(tokens)
  tokens = [w if freq[w] > 3 else '<unk>' for w in tokens]
  print(f' There are: {len(tokens)} tokens in dataset')
  return tokens

def build_dataset(words, block_size, device, train_size=0.8):
    tokens = preapare_tokens(words)

    unique_words = sorted(list(set(tokens)))
    unique_words = ['<BLOCK>'] + unique_words
    stoi = {s:i for i,s in enumerate(unique_words)}
    itos = {i:s for s,i in stoi.items()}
    vocab_size = len(stoi)
    X, Y = [], []
    context = [0] * block_size
    for w in tokens:
      ix = stoi[w]
      X.append(context)
      Y.append(ix)
      #print(' '.join(itos[i] for i in context), '----->', itos[ix])
      context = context[1:] + [ix]

    X = torch.tensor(X, device=device)
    Y = torch.tensor(Y, device=device)
    ix = torch.randperm(X.shape[0])
    X = X[ix]
    Y = Y[ix]
    n1 = int(0.8*len(tokens))
    n2 = int(0.9*len(tokens))
    Xtr, Xval, Xte = X[:n1], X[n1:n2], X[n2:]
    Ytr, Yval, Yte = Y[:n1], Y[n1:n2], Y[n2:]
    print(X.shape, Y.shape)
    return Xtr, Xval, Xte, Ytr, Yval, Yte