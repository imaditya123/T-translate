from tqdm import tqdm
import json
from collections import Counter

class Tokenizer:
  def __init__(self,lower=True,special_tokens=None):
    self.lower=lower

    if special_tokens is None:
      special_tokens=["<pad>","<bos>","<eos>","<unk>"]

    self.special_tokens=special_tokens

    self.word2idx={}
    self.idx2word={}

    self.vocab_fitted=False

  def fit(self,texts,vocab_size=None):
    tokens=[]

    for text in tqdm(texts):
      if self.lower:
        text=text.lower()

      tokens.extend(text.split())
    counter=Counter(tokens)
    # Get the most common tokens up to the specified vocab_size.
    most_common = counter.most_common(vocab_size)
    vocab_tokens = [token for token, _ in most_common]

    # Ensure special tokens come first.
    vocab = []
    for token in tqdm(self.special_tokens):
        if token not in vocab:
            vocab.append(token)
    for token in tqdm(vocab_tokens):
        if token not in vocab:
            vocab.append(token)


    self.word2idx={word:idx for idx, word in enumerate(tqdm(vocab))}
    self.idx2word={idx:word for  word,idx in tqdm(self.word2idx.items())}
    self.vocab_fitted=True

  def tokenize(self,text):
    if self.lower:
      text.lower()
    return text.split()

  def encode(self,text,add_special_tokens=True):
    tokens = self.tokenize(text)
    # Map tokens to their IDs, using the <unk> token for unknown tokens.
    token_ids = [self.word2idx.get(token, self.word2idx.get("<unk>")) for token in tokens]
    if add_special_tokens:
        token_ids = [self.word2idx.get("<bos>")] + token_ids + [self.word2idx.get("<eos>")]
    return token_ids


  def decode(self,token_ids, skip_special_tokens=True):
    tokens = [self.idx2word.get(idx, "<unk>") for idx in token_ids]
    if skip_special_tokens:
      tokens = [token for token in tokens if token not in self.special_tokens]
    return " ".join(tokens)

  def texts_to_sequences(self,texts,add_special_tokens=True):
    return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

  def sequences_to_texts(self, sequences, skip_special_tokens=True):
    return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

  @property
  def bos_token_id(self):
    return self.word2idx.get("<bos>", None)
  @property
  def eos_token_id(self):
    return self.word2idx.get("<eos>", None)
  @property
  def pad_token_id(self):
    return self.word2idx.get("<pad>", None)
  @property
  def unk_token_id(self):
    return self.word2idx.get("<unk>", None)

  def save_vocab(self, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
      json.dump(self.word2idx, f, ensure_ascii=False, indent=4)

  def load_vocab(self, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      self.word2idx = json.load(f)
    self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    self.vocab_fitted = True
