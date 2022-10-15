# import the necessary packages
from bbox_regressor import ObjectDetector
from dataset import BucketDataset
import config
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import numpy as np
import pickle
import torch
import time
import cv2
import os
	
# define normalization transforms
transforms = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=config.MEAN, std=config.STD)
])

train_dataset = BucketDataset(root_dir=config.root_dir, transform=transforms)

trainLoader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
	shuffle=True, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY, drop_last=True)

print("[INFO] total training samples: {}...".format(len(train_dataset)))

# load the ResNet50 network
resnet = resnet50(pretrained=True)
# freeze all ResNet50 layers so they will *not* be updated during the
# training process
for param in resnet.parameters():
	param.requires_grad = False

# create our custom object detector model and flash it to the current
# device
objectDetector = ObjectDetector(resnet, config.NUM_CLASSES)
objectDetector = objectDetector.to(config.DEVICE)

# define our loss functions
classLossFunc = CrossEntropyLoss()
bboxLossFunc = MSELoss()
# initialize the optimizer, compile the model, and show the model
# summary
optimizer = Adam(objectDetector.parameters(), lr=config.INIT_LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

print(objectDetector)
objectDetector.train()

# initialize a dictionary to store training history
H = {"total_train_loss": [], "total_cls_loss": [], "total_box_loss": [], "train_class_acc": []}

print("[INFO] training the network...")

for num_epoch in tqdm(range(config.NUM_EPOCHS)):
	
	totalTrainLoss = 0
	totalboxloss = 0
	totalclsloss = 0
	trainCorrect = 0
	label = 0
	steps = 0
	startTime = time.time()

	for sample in trainLoader:

		optimizer.zero_grad()

		images = sample['image'].float().to(config.DEVICE)
		bboxes = sample['bbox'].float().to(config.DEVICE)
		labels = torch.tensor([label] * config.BATCH_SIZE).to(config.DEVICE)
		
		# print('images shape ', images.shape) [32, 3, 224, 224]
		# print('bboxes shape ', bboxes.shape) [32, 1, 4]
		# print('labels shape ', labels.shape) [32]

		predictions = objectDetector(images)
		bboxLoss = bboxLossFunc(predictions[0], bboxes)
		classLoss = classLossFunc(predictions[1], labels)

		totalLoss = (config.BBOX * bboxLoss) + (config.LABELS * classLoss)
		
		totalLoss.backward()
		optimizer.step()

		totalTrainLoss += totalLoss
		totalboxloss += bboxLoss
		totalclsloss += classLoss
		steps += 1
		trainCorrect += (predictions[1].argmax(1) == labels).type(
			torch.float).sum().item()

	avgTrainLoss = totalTrainLoss / steps
	avgboxloss = totalboxloss / steps
	avgclsloss = totalclsloss / steps

	trainCorrect = trainCorrect / len(train_dataset)

	endTime = time.time()	
	time_elapsed = (endTime - startTime) / 60 #mins
	# update our training history
	H["total_train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["total_box_loss"].append(avgboxloss.cpu().detach().numpy())
	H["total_cls_loss"].append(avgclsloss.cpu().detach().numpy())
	H["train_class_acc"].append(trainCorrect)
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(num_epoch + 1, config.NUM_EPOCHS))
	print("Time: {:.3f}, Train loss: {:.6f}, Train box loss: {:.6f}, Train cls loss: {:.6f}, Train accuracy: {:.4f}".format(
		time_elapsed, avgTrainLoss, avgboxloss, avgclsloss, trainCorrect))

	scheduler.step()
	if (num_epoch + 1) % 5 == 0:
		print("[INFO] saving object detector model...")
		model_path = os.path.join(config.MODEL_DIR, str(num_epoch+1) + '.pth') 
		torch.save(objectDetector, model_path)

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["total_train_loss"], label="total_train_loss")
plt.plot(H["train_class_acc"], label="train_class_acc")
plt.title("Total Training Loss and Classification Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# save the training plot
plotPath = os.path.sep.join([config.PLOTS_PATH, "training.png"])
plt.savefig(plotPath)