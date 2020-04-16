import os
import tqdm
import argparse
import numpy as np

import torchvision.models as torch_model
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.quantize_model import *
from utils.val import validation
from train_script import train_GDFQ



parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument("-a", "--arch", type=str, default="resnet18", help="number of epochs of training")
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--n_iter", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--q_lr", type=float, default=1e-6, help="adam: learning rate")
parser.add_argument("--g_lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("-qa", "--quan_a_bit", type=int, default=4, help=" quan activation bit")
parser.add_argument("-qw", "--quan_w_bit", type=int, default=4, help=" quan weight bit")
parser.add_argument("-qb", "--quan_b_bit", type=int, default=4, help=" quan bias bit")




def main():

	args = parser.parse_args()
	valdir = os.path.join(args.data, 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	val_loader = torch.utils.data.DataLoader(
					datasets.ImageFolder(valdir,
					transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					normalize,
					])),
					batch_size=args.batch_size,
					shuffle=False,
					num_workers=4,
					pin_memory=True
				)

	criterion = nn.CrossEntropyLoss().cuda()
	FP_model = getattr(torch_model, args.arch)(pretrained=True)
	fp_1, fp_5 = validation(val_loader, FP_model, criterion)

	Q_model = quantize_model(FP_model, args.quan_a_bit, args.quan_w_bit, args.quan_b_bit)
	Q_model = freeze_bn(Q_model)
	Q_model = freeze_act(Q_model)

	q_init_1, q_init5 = validation(val_loader, Q_model, criterion)


	Q_model = un_freeze_act(Q_model)	
	Q_model = train_GDFQ.train_GDFQ(FP_model, Q_model, val_loader, criterion,
									batch_size = args.batch_size,
									total_epoch = args.n_epochs, iter_per_epoch=args.n_iter,
									q_lr = args.q_lr, g_lr = args.g_lr)
	Q_model = freeze_act(Q_model)
	q_final_1, q_final_5 = validation(val_loader, Q_model, criterion)

	print("FP Model ==> Top1: {}, Top5: {}".format(fp_1, fp_5))
	print("Q Model Initial ==> Top1: {}, Top5: {}".format(q_init_1, q_init5))
	print("Q Model Final ==> Top1: {}, Top5: {}".format(q_final_1, q_final_5))




if __name__ == "__main__":
	main()