import config
from torchvision import transforms
from dataset import BucketDataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import torch
import cv2
import os
import numpy as np
import utils

def load_model(epoch_num):

	print("[INFO] loading object detector...")
	model_path = os.path.join(config.MODEL_DIR, str(epoch_num) + '.pth')
	model = torch.load(model_path).to(config.DEVICE)
	model.eval()

	return model

# load our object detector, set it evaluation mode, and label
# encoder from disk
def load_data():

	transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),
		transforms.Normalize(mean=config.MEAN, std=config.STD)
	])

	train_dataset = BucketDataset(root_dir=config.root_dir, transform=transform, train=0)

	trainLoader = DataLoader(train_dataset, batch_size=1,
		shuffle=False, num_workers=os.cpu_count(), pin_memory=config.PIN_MEMORY)

	return trainLoader

def test(epoch_num):

	model = load_model(epoch_num)
	data_loader = load_data()
	bbox_loss_func = MSELoss()
	num_correct = 0
	total_box_loss = 0
	step = 0

	for sample in data_loader:

		images = sample['image'].float().to(config.DEVICE)
		bboxes = sample['bbox'].float().to(config.DEVICE)
		orig_image = sample['original_image'].cpu().detach().numpy()[0]

		labels = torch.tensor([0]).to(config.DEVICE)

		# print('images shape ', images.shape) #[1, 3, 224, 224]
		# print('bboxes shape ', bboxes.shape) #[1, 1, 4]
		# print('labels shape ', labels.shape) #[1]

		predictions = model(images)
		conf = torch.nn.Softmax(dim=-1)(predictions[1])
		print('conf ', conf)

		bbox_loss = bbox_loss_func(predictions[0], bboxes)
		total_box_loss += bbox_loss
		step += 1
		avg_box_loss = total_box_loss / step

		num_correct += (predictions[1].argmax(1) == labels).type(
			torch.float).sum().item()

		bbox_loss_np = bbox_loss.cpu().detach().numpy()
		avg_box_loss_np = avg_box_loss.cpu().detach().numpy()

		# display
		pred_bbox_remap = utils.remap_bbox(predictions[0], orig_image.shape)
		gt_bbox_remap = utils.remap_bbox(bboxes, orig_image.shape)

		print('gt: ', gt_bbox_remap, ' pred: ', pred_bbox_remap)
		disp_image = utils.draw_bbox(orig_image, pred_bbox_remap, color=(0, 0, 255))
		disp_image = utils.draw_bbox(disp_image, gt_bbox_remap, color=(255, 0, 0))

		cv2.imshow('disp_image ', disp_image)
		cv2.waitKey(-1)

if __name__ == "__main__":

	epoch_num = 50
	test(epoch_num)	