import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_PATIENCE = 3
MIN_DELTA = 1e-4
P_MASK = 0.15
NUM_WORKERS = 4
NUM_RUNS = 1
ALPHA_FACTOR = 1.0
WEIGHT_DECAY = 2e-5
ALPHA_REG_LAMBDA = 1e-5