import logging

# Package Metadata
__version__ = "1.0.0"
__author__ = "imaditya123"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


from .config import *
from .utils import *
import model
from .tokenizer import Tokenizer
from .dataset import TranslationDataset

logging.info("Translation model package initialized.")
