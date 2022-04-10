if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#VGG16은 3x3 conv층, 채널수는 풀링(2x2풀링사용)하면 2배 증가, FC에선 dropout, ReLU
#학습된 VGG16 사용하기
import dezero
from PIL import Image
from dezero.models import VGG16
import numpy as np

url='https://github.com/WegraLee/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path=dezero.utils.get_file(url)
img=Image.open(img_path)
x=VGG16.preprocess(img)#이미지를 행렬로
#print(type(x), x.shape)#<class 'numpy.ndarray'> (3, 224, 224)
x=x[np.newaxis]#배치용 축 추가

model=VGG16(pretrained=True)
with dezero.test_mode():
    y=model(x)
predict_id=np.argmax(y.data)

model.plot(x, to_file='vgg.pdf')
labels=dezero.datasets.ImageNet.labels()
print(labels[predict_id])
