from transformers import BertTokenizer
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 256
EPOCHS = 2
LEARNING_RATE = 3e-5
TRAINING_FILE = '.\\input\\train-v2.0.json'
VALIDATION_FILE = '.\\input\\dev-v2.0.json'
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL_PATH = '.\\model'

CONTEXT = ''
QUESTION = ''
