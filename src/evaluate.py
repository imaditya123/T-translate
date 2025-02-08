import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
import utils
import pandas as pd
from tokenizer import Tokenizer
from dataset import TranslationDataset
from model import TransformerModel


def evaluate():
    # download dataset
    df = pd.read_parquet(config.DATASET)

    # initialise tokenizer
    en_tokenizer=Tokenizer()
    kn_tokenizer=Tokenizer(lower=False)
    en_tokenizer.fit(list(df['en']),vocab_size=config.INPUT_DIM)
    kn_tokenizer.fit(list(df['kn']),vocab_size=config.OUTPUT_DIM)
    
    # Created as dataset
    translation_dataset = TranslationDataset(
                        df=df,
                        src_column='en',
                        tgt_column='kn',
                        src_tokenizer=en_tokenizer,
                        tgt_tokenizer=kn_tokenizer,
                        src_max_len=256,
                        tgt_max_len=256
                    )
    # Dataloader
    data_loader = DataLoader(translation_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Create the model instance.
    model = TransformerModel(config.INPUT_DIM, 
                             config.OUTPUT_DIM, 
                             config.D_MODEL, 
                             config.NUM_HEADS,
                             config.NUM_ENCODER_LAYERS, 
                             config.NUM_DECODER_LAYERS, 
                             config.HIDDEN_DIM, 
                             config.DROPOUT)
    
    model = model.to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    utils.load_model(model,optimizer,os.path.join(config.SAVED_MODEL,"en_kn_model.pth"),config.DEVICE)
    pad_token_id = 0
    avg_loss, avg_accuracy = evaluate(model, data_loader, criterion, config.DEVICE,pad_token_id)


if __name__ == "__main__":
    evaluate()