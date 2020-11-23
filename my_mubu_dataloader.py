import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from my_mubugen import MubuGen


# from img_augment import letter_img

class ImgCombineDataset(Dataset):
    def __init__(self, classPath, imgDir, inputDim, randomP=0.5, image_weights=False, rect=False, use_iaa=True, is_train=True):
        self.getpic = MubuGen('tmp_mubu')
        self.clsList = load_classes(classPath)
        # print(classPath, self.clsList)
        self.bgDir = os.path.join(imgDir, 'background')
        self.lgDir = os.path.join(imgDir, 'logo3')
        self.nlgDir = os.path.join(imgDir, 'nologo')
        self.imgBgPathList = []
        if is_train:
            self.BgDirNameList = ['class','tv']  # ['class','pic','tv','bigtv']
        else:
            self.BgDirNameList = ['room']
        for name in self.BgDirNameList:
            self.imgBgPathList += glob.glob(os.path.join(self.bgDir, name, '*.jpg'))
        # print(self.imgBgPathList)
        # self.imgBgPathList = glob.glob(os.path.join(self.bgDir,'*.jpg'))
        self.imgLgPathList = glob.glob(os.path.join(self.lgDir, '*.jpg'))
        self.imgNLgPathList = glob.glob(os.path.join(self.nlgDir, '*.jpg'))
        self.inputDim = inputDim
        self.randomP = randomP

        self.labels = [np.zeros((0, 5))] * (len(self.imgBgPathList))
        print(len(self.imgBgPathList))

        self.image_weights = image_weights
        self.rect = False if image_weights else rect

        self.use_iaa = use_iaa

        self.myiaa = MyIaa()

    def __len__(self):
        return len(self.imgBgPathList)

    def __getitem__(self, idx, direct_output=False):
        imgBgPath = self.imgBgPathList[idx]
        imgBg = cv2.imread(imgBgPath)
        # imgBg, _= letter_img(imgBg, self.inputDim, fillValue=128)
        imgBg = cv2.resize(imgBg, (512, 512))
        bboxes = []
        topNum = random.randint(1, 3)
        for lg_idx in range(topNum):
            imgLg = self.getpic.gen()
            cls = 0

            if lg_idx == 0:
                bottom_iaa = True
                merge_iaa = False
            else:
                bottom_iaa = False
                merge_iaa = True
            imgBg, bboxes = self.myiaa.output(imgLg, cls, imgBg, bboxes=bboxes, bottom_iaa=bottom_iaa,
                                              merge_iaa=merge_iaa)
            # imgBg, _= letter_img(imgBg, self.inputDim)
            imgBg = cv2.resize(imgBg, (512, 512))

        # imgBg = cv2.cvtColor(imgBg, cv2.COLOR_BGR2GRAY)
        # imgBg = cv2.cvtColor(imgBg, cv2.COLOR_GRAY2BGR)
        # imgBg = self.histogram_equalization(imgBg)

        if direct_output == True:
            return imgBg, bboxes
        else:
            # labels = np.zeros((bboxes.shape[0],5))#np.column_stack((np.zeros((labels.shape[0],1)),labels))
            # labels[:,1:]=bboxes
            labels = bboxes

            labels = np.array(labels).astype(float)
            nL = len(labels)
            if nL:
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= imgBg.shape[0]  # height
                labels[:, [1, 3]] /= imgBg.shape[1]  # width
            labels_out = torch.zeros((nL, 6))
            if nL:
                labels_out[:, 1:] = torch.from_numpy(labels)
            # Normalize
            imgBg = imgBg[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            imgBg = np.ascontiguousarray(imgBg, dtype=np.float32)  # uint8 to float32
            # imgBg /= 255.0  # 0 - 255 to 0.0 - 1.0
            imgTensor = torch.from_numpy(imgBg)
            # print(labels_out, imgBg.shape, imgLg.shape)
            return imgTensor, labels_out, imgBgPath, imgBg.shape

    @staticmethod
    def collate_fn(batch):
        img, label, path, hw = list(zip(*batch))  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, hw

    def histogram_equalization(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(10, 10))
        equ = clahe.apply(img)
        equ = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
        return equ


def letter_img(img, inputDim=416):
    h0, w0 = img.shape[:2]  # orig hw
    r = inputDim / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        # interp = cv2.INTER_AREA if r < 1 and not inputDim else cv2.INTER_LINEAR
        img_new = cv2.resize(img, (int(w0 * r), int(h0 * r)))
    return img_new, (h0, w0, img.shape[:2])  # img, hw_original, hw_resized


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

class MyIaa(object):
    def __init__(self):
        self.seq0 = iaa.Sequential([
            iaa.Fliplr(0.5),
            # iaa.OneOf([iaa.AddToBrightness(add=(-50, 50), from_colorspace='BGR'),
            #            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-50, 50), from_colorspace='BGR'),
            #            iaa.MultiplyBrightness((0.5, 1.5), from_colorspace='BGR'),
            #            # iaa.Multiply(mul=(0.5, 1.5), per_channel=True),
            #            ]),

            # iaa.Add(10, per_channel=True),
            iaa.SomeOf((0, 1), [iaa.TranslateY(percent=(-0.2, 0.1)),
                                ], random_order=True),

        ])
        self.seq1 = iaa.Sequential([iaa.Fliplr(0.5),
                                    iaa.SomeOf((0, 5), [iaa.TranslateX(percent=(-0.1, 0.1)),
                                                        iaa.TranslateY(percent=(-0.1, 0.1)),
                                                        iaa.Rotate((-10, 10)),
                                                        iaa.ShearX((-10, 10)),
                                                        iaa.ShearY((-10, 10)),
                                                        # iaa.AddToBrightness(add=(-50,50),from_colorspace='BGR'),
                                                        ], random_order=True),
                                    ], random_order=True)
        self.seq2 = iaa.Sequential([  # iaa.Fliplr(0.5),
            iaa.SomeOf((1, 1), [iaa.OneOf([iaa.MotionBlur(k=(3, 3)),
                                           iaa.GaussianBlur((0, 3)),
                                           iaa.AverageBlur(k=(3, 3)),
                                           iaa.MedianBlur(k=(3, 3)),
                                           ]),
                                iaa.OneOf([iaa.AddElementwise((-10, 10), per_channel=0.5),
                                           iaa.AdditiveGaussianNoise(loc=0, scale=0.01 * 255, per_channel=0.5),
                                           iaa.AdditiveLaplaceNoise(scale=0.01 * 255, per_channel=0.5),
                                           iaa.AdditivePoissonNoise(10, per_channel=0.5),
                                           ]),
                                # iaa.TranslateX(percent=(-0.1, 0.1)),
                                # iaa.TranslateY(percent=(-0.1, 0.1)),
                                iaa.AddToBrightness(add=(-50,50),from_colorspace='BGR'),
                                ], random_order=True),
        ], random_order=True)

    def iaa_img(self, img, seq, bboxes=[]):
        imgs = np.expand_dims(img, 0)
        if len(bboxes) > 0:
            bboxes = np.array([bboxes])
            imgs, bboxes[:, :, 1:] = seq(images=imgs, bounding_boxes=bboxes[:, :, 1:])
            return imgs[0], bboxes[0]
        else:
            imgs = seq(images=imgs)
            return imgs[0], bboxes

    def merge(self, imgt, cls, imgb, bboxes=[], tryTimes=10):
        bboxes = np.array(bboxes)
        ht, wt, _ = imgt.shape  # top
        hb, wb, _ = imgb.shape  # bottom
        randomScale = random.random()
        if randomScale < 0.3:
            scale = random.uniform(0.1, 0.2)
        elif randomScale < 0.9:
            scale = random.uniform(0.2, 0.6)
        else:
            scale = random.uniform(0.6, 0.9)

        scale_fit = min(hb/ht, wb/wt)
        wtr = int(wt * scale * scale_fit)
        htr = int(ht * scale * scale_fit)
        imgtr = cv2.resize(imgt, (wtr, htr), interpolation=cv2.INTER_AREA)
        for i in range(tryTimes):
            x0 = random.randint(0, wb - wtr)
            y0 = random.randint(0, hb - htr)
            bboxNew = np.array([cls, x0, y0, x0 + wtr, y0 + htr])
            if len(bboxes) == 0:
                break
            for bbox in bboxes:
                iou = self.cal_iou(bbox[1:], bboxNew[1:])
                if iou > 0:
                    break
            if iou <= 0:
                break
            if i == tryTimes - 1:
                return imgb, bboxes
        # bboxes.append(bboxNew)
        if len(bboxes) == 0:
            bboxes = np.array([bboxNew])
        else:
            bboxes = np.row_stack([bboxes, bboxNew])
        # bright = random.uniform(0.7, 1.1)
        imgbCut = imgb[y0:y0 + htr, x0:x0 + wtr]
        imgtr[imgtr == 0] = imgbCut[imgtr == 0]
        # imgb[y0:y0 + htr, x0:x0 + wtr] = cv2.addWeighted(imgtr, bright, imgbCut, 1 - bright, 0)
        imgb[y0:y0 + htr, x0:x0 + wtr] = imgtr
        return imgb, bboxes

    def draw_bbox(self, img0, bbox, text=None, color=(0, 255, 0)):
        if bbox == []:
            return img0
        img = img0.copy()
        minX, minY, maxX, maxY = bbox
        cv2.rectangle(img, (minX, minY), (maxX, maxY), color, 2)
        if text != None:
            cv2.putText(img, text, (minX, minY), cv2.LINE_AA, 1, color, 2)
        return img

    def cal_iou(self, bbox1, bbox2):
        cx1, cy1, cx2, cy2 = bbox1
        gx1, gy1, gx2, gy2 = bbox2

        carea = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)  # C\B5\C4\C3\E6\BB\FD
        garea = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)  # G\B5\C4\C3\E6\BB\FD

        x1 = max(cx1, gx1)
        y1 = max(cy1, gy1)
        x2 = min(cx2, gx2)
        y2 = min(cy2, gy2)
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)
        area = w * h  # C\A1\C9G\B5\C4\C3\E6\BB\FD
        iou = area / (carea + garea - area)
        return iou

    def output(self, imgt, cls, imgb, bboxes=[], bottom_iaa=True, merge_iaa=True):
        # bboxes=[]
        if bottom_iaa:
            imgb, bboxes = self.iaa_img(imgb, self.seq1, bboxes=bboxes)
        imgt, _ = self.iaa_img(imgt, self.seq0)
        imgb, bboxes = self.merge(imgt, cls, imgb, bboxes=bboxes)
        if merge_iaa:
            imgb, bboxes = self.iaa_img(imgb, self.seq2, bboxes=bboxes)
        return imgb, bboxes


if __name__ == '__main__':
    from tqdm import tqdm
    from torchvision import transforms
    import matplotlib.pyplot as plt
    from utils.utils import plot_one_box
    import math


    def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
        tl = 3  # line thickness
        tf = max(tl - 1, 1)  # font thickness
        if os.path.isfile(fname):  # do not overwrite
            return None

        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        # un-normalise
        if np.max(images[0]) <= 1:
            images *= 255

        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs ** 0.5)  # number of subplots (square)

        # Check if we should resize
        scale_factor = max_size / max(h, w)
        if scale_factor < 1:
            h = math.ceil(scale_factor * h)
            w = math.ceil(scale_factor * w)

        # Empty array for output
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

        # Fix class - colour map
        prop_cycle = plt.rcParams['axes.prop_cycle']
        # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
        hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

        for i, img in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break

            block_x = int(w * (i // ns))
            block_y = int(h * (i % ns))

            img = img.transpose(1, 2, 0)
            if scale_factor < 1:
                img = cv2.resize(img, (w, h))

            mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
            if len(targets) > 0:
                image_targets = targets[targets[:, 0] == i]
                boxes = xywh2xyxy(image_targets[:, 2:6]).T
                classes = image_targets[:, 1].astype('int')
                gt = image_targets.shape[1] == 6  # ground truth if no conf column
                conf = None if gt else image_targets[:, 6]  # check for confidence presence (gt vs pred)

                boxes[[0, 2]] *= w
                boxes[[0, 2]] += block_x
                boxes[[1, 3]] *= h
                boxes[[1, 3]] += block_y
                for j, box in enumerate(boxes.T):
                    cls = int(classes[j])
                    color = color_lut[cls % len(color_lut)]
                    cls = names[cls] if names else cls
                    if gt or conf[j] > 0.3:  # 0.3 conf thresh
                        label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                        plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

            # Draw image filename labels
            if paths is not None:
                label = os.path.basename(paths[i])[:40]  # trim to 40 char
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220],
                            thickness=tf,
                            lineType=cv2.LINE_AA)

            # Image border
            cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

        if fname is not None:
            mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

        return mosaic


    def load_classes(path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)


    unloader = transforms.ToPILImage()

    dataset = ImgCombineDataset(classPath='class.txt',
                                imgDir='../mydata/train',
                                inputDim=416,
                                randomP=0.5, image_weights=False, rect=False, use_iaa=True)
    trainloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=4,
                                              num_workers=8,
                                              shuffle=True,  # Shuffle=True unless rectangular training is used
                                              pin_memory=True,
                                              collate_fn=dataset.collate_fn)

    # for batch_idx, (imgs, targets, paths, wh) in enumerate(trainloader):
    nb = len(trainloader)
    pbar = tqdm(enumerate(trainloader), total=nb)
    for i, (imgs, targets, paths, wh) in pbar:
        #     print(imgs, targets, wh,'\n')
        #     print(targets.shape,imgs.shape)

        # img1, img2 = imgs.chunk(2, dim=0)
        # image = img1.squeeze(0)
        # image = unloader(image)
        # plt.imshow(image)
        # plt.figure()
        # image = img2.squeeze(0)
        # image = unloader(image)
        # plt.imshow(image)

        imgs = imgs.float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        names = load_classes('class.txt')  # class names
        plot_images(imgs, targets, paths=paths, names=names, fname='test1.jpg')
        tmpimg = cv2.imread('test1.jpg')
        plt.imshow(tmpimg)
        os.remove('test1.jpg')

        plt.pause(0.001)
        print('!!!')

