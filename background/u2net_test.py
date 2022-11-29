import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
import glob

from background.data_loader import RescaleT
from background.data_loader import ToTensorLab
from background.data_loader import SalObjDataset

import cv2

from background.model import U2NET  # full size version 173.6 MB

# normalize the predicted SOD probability map

model_name = 'u2net'  # u2netp
model_dir = os.getcwd() + '/background/saved_models/' + model_name + '.pth'
net = None

def removeBg(image,species):
    if(species == "corn"):
        return image
    path = os.getcwd() + "/background/src/image.jpeg"
    image.save(path)
    return removeBgColor()

def loadModel():
    global net
    if (model_name == 'u2net'):
        net = U2NET(3, 1)
    net.load_state_dict(torch.load(
        model_dir, map_location=torch.device('cpu')))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def applyMaskToImg(mask, org_img):

    mask = cv2.cvtColor(src=mask, code=cv2.COLOR_BGR2RGB)

    org_img = cv2.cvtColor(src=cv2.imread(
        filename=org_img), code=cv2.COLOR_BGR2RGB)

    img_cpy = org_img.copy()

    img_cpy[mask > 200] = 0

    img = org_img - img_cpy

    return img


def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')

    img_name = image_name.split("/")[-1]

    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]),
                    resample=Image.Resampling.BILINEAR)

    mask = np.array(imo)
    org_img = np.asarray(Image.open(image_name))

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    cut_out_img = applyMaskToImg(mask=mask, org_img=image_name)
    img=cv2.cvtColor(cut_out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename=str(d_dir+imidx+'.png'),
                img=cv2.cvtColor(cut_out_img, cv2.COLOR_RGB2BGR))
    return img



def removeBgColor():

    # --------- 1. get image path and name ---------
    image_dir = os.getcwd() + '/background/src/'

    prediction_dir = os.getcwd() + '/background/result/'


    img_name_list = glob.glob(image_dir + '*')

    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)



    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):


        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]

        pred = normPRED(pred)

        # save results to test_results folder
        return save_output(img_name_list[i_test], pred, prediction_dir)


        
