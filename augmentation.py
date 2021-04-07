import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
from RandAugment import RandAugment


def configure_transform(phase: str, transform_type: str):
    if phase == "train":
        if transform_type == "base":
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384), Image.BILINEAR),
                    transforms.CenterCrop((384, 384)),
                    transforms.RandomResizedCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif transform_type == 'facecrop':
            transform = transforms.Compose(
                [
                    transforms.Lambda(lambda x: crop(x)),
                    transforms.Resize((312, 234)),
                    transforms.RandomCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif transform_type == 'random':
            transform  = transforms.Compose(
                [
                    transforms.CenterCrop((384, 384)),
                    transforms.RandomResizedCrop((224, 224)),
                    RandAugment(2, 9), # N: 몇 개 선택할지 M: 몇 번 변화시킬 것인지
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        else:
            raise NotImplementedError()

    else:
        if transform_type == "base":
            transform = transforms.Compose(
                [
                    transforms.Resize((512, 384), Image.BILINEAR),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif transform_type == 'facecrop':
            transform = transforms.Compose(
                [
                    transforms.Lambda(lambda x: crop(x)),
                    transforms.Resize((312, 234)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        elif transform_type == 'random':
            transform  = transforms.Compose(
                [
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        else:
            raise NotImplementedError()
            
    return transform


def crop(img, scale: int=64, p=0.5, rescaled=True):
    """
    Description of crop

    Args:
        img (undefined):
        scale (int=64):
        p=0.5 (undefined):
        rescaled=True (undefined):

    """
    if np.random.uniform(0, 1) >= p: # 시행하지 않을 경우
        return img
        
    transform = transforms.Compose([transforms.Resize((int(scale*(4/3)), scale)), transforms.Grayscale()])
    img_resized = np.array(transform(img))
    MAX_H, MAX_W = np.array(img).shape[:2]

    vline = get_vline(img_resized, scale)
    h_upper, h_lower = get_hline(img_resized, vline)

    if rescaled: # 입력할 때의 이미지 크기에 맞게 수직선의 위치를 조정
        origin_v_scale = np.array(img).shape[1]
        origin_h_scale = np.array(img).shape[0]

        vline = int(origin_v_scale * (vline / scale))
        h_lower = int(origin_h_scale * (h_lower / int(scale*(4/3))))
        h_upper = int(origin_h_scale * (h_upper / int(scale*(4/3))))

    hline = int((h_lower + h_upper) / 2)
    height = abs(h_upper - h_lower)
    width = int(height * (3/4))

    coord_y = np.clip([hline-height, hline+height], 0, MAX_H)
    coord_x = np.clip([vline-width, vline+width], 0, MAX_W)
    cropped = np.array(img)[coord_y[0]:coord_y[1], coord_x[0]:coord_x[1], :]

    return Image.fromarray(cropped)


def get_similarity(pixel_dist_left, pixel_dist_right):
    """
    픽셀 분포 간 유사도를 측정하는 함수
    분포 간 intersection을 유사도로 가정하여 높을 수록 유사도가 높다
    """
    hist_left = cv2.calcHist(images=[pixel_dist_left], channels=[0], mask=None, histSize=[32], ranges=[0,256])
    hist_right = cv2.calcHist(images=[pixel_dist_right], channels=[0], mask=None, histSize=[32], ranges=[0,256])
    similarity = cv2.compareHist(H1=hist_left, H2=hist_right, method=cv2.HISTCMP_INTERSECT)
    return similarity

def split_image(img: np.array, v, scale):
    """수직선을 기준으로 이미지를 분할
    좌우로 이미지를 분할했을 때, 가로 길이가 같도록 가로 길이를 조정
    
    Args
    ---
    img: np.array, grayscale
    v: 수직선의 위치
    scale: crop 과정에 활용된 scale
    """
    if v >= scale:
        margin = scale - v%scale
        pixel_dist_left = img[:, v-margin:v]
        pixel_dist_right = img[:, v:]
    else:
        margin = v%scale
        pixel_dist_left = img[:, :v]
        pixel_dist_right = img[:, v:v+margin]
    return pixel_dist_left, pixel_dist_right


def get_vline(img_resized: np.array, scale: int=64):
    v_list = [i+int(scale*(2/5)) for i in range(scale//5)][::2]
    importances = np.zeros(len(v_list))

    for idx, v in enumerate(v_list):
        pixel_dist_left, pixel_dist_right = split_image(img_resized, v, scale)
        similarity = get_similarity(pixel_dist_left, pixel_dist_right)
        importances[idx] += (similarity / (v*scale)) # normalize
    v = v_list[np.argsort(importances)[3]] # (경험적) importance가 가장 높은 값이 아닌 2~3번째로 높은 값이 원하는 수직선이 나오는 경우가 많음
    return v

def get_hline(img: np.array, v):
    UPPER_QUANTILE = 0.35
    LOWER_QUANTILE = 1 - UPPER_QUANTILE
    FACTOR = 8

    v0 = moving_average(img[:, v], FACTOR) # 단순이동평균
    v1 = np.roll(v0, shift=1, axis=0)
    sma_interval = v0.shape[0]
    offset = int(sma_interval * LOWER_QUANTILE)

    h_upper = np.argmax(np.abs(v0[1:int(sma_interval*UPPER_QUANTILE)] - v1[1:int(sma_interval*UPPER_QUANTILE)])) + FACTOR - 1
    h_lower = np.argmax(np.abs(v0[int(sma_interval*LOWER_QUANTILE):-1] - v1[int(sma_interval*LOWER_QUANTILE):-1])) + offset - FACTOR + 1

    return h_upper, h_lower

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w





