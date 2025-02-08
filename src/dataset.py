
import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, df, src_column, tgt_column, src_tokenizer, tgt_tokenizer,
                 src_max_len=20, tgt_max_len=20):
      self.df = df.reset_index(drop=True)
      self.src_column = src_column
      self.tgt_column = tgt_column
      self.src_tokenizer = src_tokenizer
      self.tgt_tokenizer = tgt_tokenizer
      self.src_max_len = src_max_len
      self.tgt_max_len = tgt_max_len

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
      # Get the source and target text for the given index
      src_text = self.df.loc[idx, self.src_column]
      tgt_text = self.df.loc[idx, self.tgt_column]

      # Encode the texts (the encode() method automatically adds <bos> and <eos> tokens)
      src_ids = self.src_tokenizer.encode(src_text)
      tgt_ids = self.tgt_tokenizer.encode(tgt_text)

      # Pad or truncate the sequences to fixed lengths
      src_ids = self.pad_or_truncate(src_ids, self.src_max_len, self.src_tokenizer.pad_token_id)
      tgt_ids = self.pad_or_truncate(tgt_ids, self.tgt_max_len, self.tgt_tokenizer.pad_token_id)

      # Convert lists to tensors
      src_tensor = torch.tensor(src_ids, dtype=torch.long)
      tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)

      return src_tensor, tgt_tensor

    def pad_or_truncate(self, token_ids, max_len, pad_token_id):
        """
        Pads token_ids to max_len if shorter, or truncates if longer.
        """
        if len(token_ids) < max_len:
            token_ids = token_ids + [pad_token_id] * (max_len - len(token_ids))
        else:
            token_ids = token_ids[:max_len]
        return token_ids
