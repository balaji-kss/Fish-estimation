import torch
import os
# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
# BASE_PATH = "dataset"
# IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
# ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])
# define the path to the base output directory
BASE_OUTPUT = "./output/"
# define the path to the output model, label encoder, plots output
# directory, and testing image paths
MODEL_DIR = os.path.sep.join([BASE_OUTPUT, "models"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
root_dir = '/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/train/data/'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False
# specify ImageNet mean and standard deviation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-3
NUM_EPOCHS = 50
NUM_CLASSES = 2
BATCH_SIZE = 8
# specify the loss weights
LABELS = 1.0
BBOX = 1.0

##### TEST #####
FILL_MODEL_PATH = "/home/balaji/Documents/code/RSL/Fish/Fish-estimation/bucket_detection/train_rcnn/fill_estimation/output/models/50.pth"
FILL_NUM_CLASSES = 1