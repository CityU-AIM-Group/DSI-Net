
#data
DATA_ROOT = '/home/meiluzhu2/data/WCE/WCE_larger'
BATCH_SIZE = 8
NUM_WORKERS = 8
DROP_LAST = True
SIZE = 256

#training
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 1e-5
NUM_CLASSES_CLS = 3
TRAIN_NUM = 2470
EPOCH = 200
STEPS = (TRAIN_NUM/BATCH_SIZE)*EPOCH
FP16 = False
VERBOSE = False
SAVE_PATH = 'checkpoints/'
LOG_PATH = 'logs/'
COLOR = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'purple', 'pink','peru']

#network
INTERMIDEATE_NUM = 64
OS = 8
EM_STEP = 3
##gumbel
GUMBEL_FACTOR = 1.0
GUMBEL_NOISE = True



