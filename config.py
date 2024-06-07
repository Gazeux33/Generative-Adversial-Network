import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NB_CHANNEL = 3
Z_DIM = 128
EPOCHS = 20
LAMBDA = 10
N_CRITIC = 4
LEARNING_RATE = 0.0002
MODEL_DIR = "models"
RESULT_DIR = "results"
