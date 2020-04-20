import os
import tqdm
import argparse
import numpy as np

import torchvision.models as torch_model
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from train_script import train_GDFQ, train_zeroQ
from utils.val import validation
from utils.quantize_model import *
from pytorchcv.model_provider import get_model as ptcv_get_model


parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument("-a", "--arch", type=str,
                    default="resnet18", help="number of epochs of training")
parser.add_argument("-m", "--method", type=str, default="GDFQ",
                    help="method of training")
parser.add_argument("--n_epochs", type=int, default=400,
                    help="number of epochs of training")
parser.add_argument("--n_iter", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="size of the batches")
parser.add_argument("--q_lr", type=float, default=1e-6,
                    help="adam: learning rate")
parser.add_argument("--g_lr", type=float, default=1e-3,
                    help="adam: learning rate")
parser.add_argument("-qa", "--quan_a_bit", type=int,
                    default=4, help=" quan activation bit")
parser.add_argument("-qw", "--quan_w_bit", type=int,
                    default=4, help=" quan weight bit")
parser.add_argument("-qb", "--quan_b_bit", type=int,
                    default=4, help=" quan bias bit")


def main():

    args = parser.parse_args()

    # restrice method input
    assert args.method in ["zeroQ", "GDFQ"]
    
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
	# prepare validation data
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

    for_inception = args.arch.startswith('inception')
    
    # FP_model = getattr(torch_model, args.arch)(pretrained=True)
    FP_model = ptcv_get_model(args.arch, pretrained=True)
    # fp_1, fp_5 = validation(val_loader, FP_model)

    Q_model = quantize_model(FP_model, args.quan_a_bit,
                             args.quan_w_bit, args.quan_b_bit)
    
	
    # _, _ = validation(val_loader, Q_model, criterion)
    
    # exit()

    if "GDFQ" == args.method:
        Q_model = train_GDFQ.train_GDFQ(FP_model, Q_model, val_loader,
                                        batch_size=args.batch_size,
                                        total_epoch=args.n_epochs, iter_per_epoch=args.n_iter,
                                        q_lr=args.q_lr, g_lr=args.g_lr,
                                        for_incep=for_inception)

    elif "zeroQ" == args.method:

        Q_model = train_zeroQ.train_zeroQ(FP_model, Q_model, 
                                          val_loader,
                                          batch_size=args.batch_size,                                          
                                          for_incep=for_inception)
    exit()
    Q_model = freeze_act(Q_model)
    q_final_1, q_final_5 = validation(val_loader, Q_model)

    # print("FP Model ==> Top1: {}, Top5: {}".format(fp_1, fp_5))
    print("Q Model Final ==> Top1: {}, Top5: {}".format(q_final_1, q_final_5))


if __name__ == "__main__":
    main()
