import sys
import os.path
import cv2
import numpy as np
import torch
from ColabESRGAN import architecture
import math
from PIL import Image
import io


class EsrGan:
    # hw = cpu, or cuda
    def __init__(self, model_path=None):
        hw = 'cpu'
        if torch.cuda.is_available():
            # 使用cuda加速
            hw = 'cuda'
        if hw == 'cpu':
            self.device = torch.device('cpu')
        if hw == 'cuda':
            self.device = torch.device('cuda')
        self.model = architecture.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', 
                                           mode='CNA', res_scale=1, upsample_mode='upconv')
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False
        self.model = self.model.to(self.device)
        print('Model warmup complete')

    def run_esrgan(self, img, mosaic_res=1):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)

        # image to device
        img_LR = img_LR.to(self.device)

        output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()

        _, im_buf_arr = cv2.imencode(".png", output)
        image = Image.open(io.BytesIO(im_buf_arr))
        # 将图像转换为NumPy数组
        img = np.array(image)

        # img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return img
