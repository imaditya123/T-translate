import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_DIM = 8000   # Vocabulary size for source language
OUTPUT_DIM = 8192  # Vocabulary size for target language
BATCH_SIZE=32
D_MODEL = 512
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 6
HIDDEN_DIM = 1024
DROPOUT = 0.1

LEARNING_RATE=3e-4

DATASET="hf://datasets/Hemanth-thunder/english-to-kannada-mt/data/train-00000-of-00001.parquet"

SAVED_MODEL="models"