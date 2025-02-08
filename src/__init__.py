import logging

# Package Metadata
__version__ = "1.0.0"
__author__ = "Your Name"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


from .config import *
from .utils import *
from .model import TransformerModel
from .tokenizer import Tokenizer
from .dataset import TranslationDataset

logging.info("Translation model package initialized.")
