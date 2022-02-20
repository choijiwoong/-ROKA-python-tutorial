"""
Captum은 PyTorch기반 구축된 모델 해석을 위한 확장가능한 오픈소스 라이브러리이다. 이는 통합 gradient를 포함한 최신 알고리즘을 제공하여 모델 출력에 기여하는 기능을 쉽게 이해할 수 있게 해준다.
Captum의 3가지 주요 속성은 Feature Attribution(입력의 기능측면), Layer Attribution(히든레이어활동 조사), Neuron Attribution(단일 뉴런의 활동에 중점)이다.
또한 시각화해주는 향상된 도구인 captum.attr.visualization모듈(이미지속성 시각화), captum Insights(미리 만들어진 시각화가 포함된 시각화 위젯을 제공)를 제공한다.
 특정 속성은 특정 출력을 입력의 특성에 귀속시킨다. Integrated Gradients는 captum에서 지원하는 기능속성 알고리즘으로, 모델 출력 기울기의 적분을
근사화하여 각 입력 기능에 중요도 점수를 할당한다. Layer attribution으로는 hiddenlayer활동을 입력기능에 귀속할 수 있으며, 모델 내 컨볼루션
레이어중 하나의 활동을 조사한다..
"""
#[First Example]
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

import captum
from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import os, sys
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

model=models.resnet101(pretrained=True)#download pretrained ResNet model by torchvision
model=model.eval()#set eval mode because we don't train now

test_img=Image.open('img/common.jpg')
test_img_data=np.asarray(test_img)#load image as nparray
plt.imshow(test_img_data)
plt.show()

#model expects 550x392 3-color image
transform=transforms.Compose([transforms.Resize(550), transforms.CenterCrop(392), transforms.ToTensor()])#set transform
transform_normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#set normalize

transformed_img=transform(test_img)#do transform
input_img=transform_normalize(transformed_img)#do normalize
input_img=input_img.unsqueeze(0)#do unaqueeze

labels_path='img/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels=json.load(json_data)#load json to idx_to_labels

#question
output=model(input_img)
output=F.softmax(output, dim=1)
prediction_score, pred_label_idx=torch.topk(output, 1)#가장 큰 요소 반환
pred_label_idx.squeeze_()
predicted_label=idx_to_labels[str(pred_label_idx.item())][1]
print('Predicted: ', predicted_label, '(', prediction_score.squeeze().item(), ')')#lakeside


#[Feature Attribution with Integrated Gradients]_뭔진 모르겠지만 핵심 위치 주변에서 가장 강한신호를 볼 수 있다네.. 위에 resnet이 판단하는 핵심 기능인듯..
integrated_gradients=IntegratedGradients(model)#Initialization of attribution algorithm by model
attributions_ig=integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)#get attribute of input_image

#show oritinal image for comparision
_=viz.visualize_image_attr(None, np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)), method="original_image", title="Original Image")

default_cmap=LinearSegmentedColormap.from_list('custom blue', [(0, '#ffffff'), (0.25, '#0000ff'), (1, '#0000ff')], N=256)

#show changed image
_=viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                           np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                           method='heat_map',
                           cmap=default_cmap,
                           show_colorbar=True,
                           sign='positive',
                           title='Integrated Gradients')

#[Feature Attribution with Occlusion]_얘도 마찬가지고..
occlusion=Occlusion(model)
attributions_occ=occlusion.attribute(input_img, target=pred_label_idx, strides=(3,8,8), sliding_window_shapes(3, 15, 15), baselines=0)
_=viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                    ["original_image", "heat_map", "heat_map", "masked_image"],
                                    ["all", "positive", "negative", "positive"],
                                    show_colorbar=True,
                                    titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
                                    fig_size=(18,6))

#[Layer Attribution with Layer GradCAM]
layer_gradcam=LayerGradCam(model, model.layer3[1].conv2)#GrandCam은 레이어의 출력기울기를 계산하고 각 출력채널의 평균을 구하고, 각 채널의 평균 기울기에 레이어 활성화를 곱한다.
attributions_lgc=layer_gradcam.attribute(input_img, target=pred_label_idx)#검사하려는 은닉 레이어를 지정해야한다. attribute()로!
_=viz.visualize_image_attr(attributions_lgc[0].cpu().permute(1,2,0).detach().numpy(), sign="all", title="Layer 3 Block 1 Conv 2")
#아쉽게도 실행이 안되서 잘은 모르지만 은닉 레이어의 속성?을 조사하기 위한 일련의 과정인듯. 위는 시각화 과정까지 포함시키고.

#LayerAttribution의 interpolate()를 사용해서 속성데이터를 upsampling하여 입력 이미지와 비교가 가능하다는데 뭔진 모르겠고 비교하는 시각화 과정인듯
upsamp_attr_lgc=LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])
print(attributions_lgc.shape)
print(upsamp_attr_lgc.shape)
print(input_img.shape)
_=viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
                                    transformed_img.permute(1,2,0).numpy(),
                                    ["original_image", "blended_heat_map", "masked_image"],
                                    ["all", "positive", "positive"],
                                    show_colorbar=True,
                                    titles=["Original", "Positive Attribution", "Masked"],
                                    fig_size=(18,6))#d역시 숨겨진 레이어가 입력에 어떻게 반응하는지를 시각화해준다고 한다.
"""
약간 중간점검느낌으로 서칭 해보니 Captum의 visualization함수는 matplotlib의 figure로 이미지를 표시하고, visualize_image_attr_multiple은 원본이미지를 함께 보여준다.
heat map에서 positive옵션을 주면 target클래스에 대한 활성화 속성을 보여주고, nagative를 주면 target이외의 클래스에 대한 활성을 보여준다고 한다.
즉, 이미지 활성도를 target, non-target으로 나누어 볼 수 있는옵션이라는 듯. 우선 Captum으로 속성을 계산하는 방법으로 Occlusion이나, Integrated Gradient,
Deconvolution등이 있는거고 시각화는 visualization유틸로 다 하는듯. 그럼 속성이 무엇인지만 알면 간략한 이해가 완성될 것 같은데, 위에 정리한것처럼
Feature Attribution, Layer Attribution, Neuron Attribution로 분류되고, 모델의 출력에 대한 각 입력기능의 기여도 평가, 레이어의 각 뉴련의 기여도, 숨겨진 뉴런의 활성화에 대한 입력기능의 기여도
등 전반적인 기여도를 평가하는 것으로 보아 실질적인 training보다는 그걸 구성하는 low level기능으로 판단. 모든 속성들이 의미하는 것은 기여도이며
이 기여도가 입력기능인지, 레이어인지, 숨겨진뉴런활성화인지 등으로 나뉘어 있는듯. 그리고 위의 예들은 여러 attribution등을 위의 사진판단하는 resnet에
적용시켜본거고 어떤 attribution이든지간에 상관없이 기여도가 높은부분을 시각화해보니 비슷한 분포를 가지더라 라는 의미인듯!

"""

#[Visualization with Captum Insight] 이는 시각화 위젯으로 여러 이미지 분류 추론을 시각화할 수 있다.
imgs=['img/cat.jpg', 'img.teapot.jpg', 'img/trilobite.jpg']

for img in imgs:#just common way for predict
    img=Image.open(img)#open
    transformed_img=transform(img)#transform
    input_img=transform_normalize(transformed_img)#normalize

    output=model(input_img)#get predict
    output=F.softmax(output, dim=1)#activated
    prediction_score, pred_label_idx=torch.topk(output, 1)#get top element
    pred_label_idx.squeeze_()#zip
    predicted_label=idx_to_labels[str(pred_label_idx.item())][1]#get label information of predict
    print('Predicted: ', predicted_label, '/', pred_label_idx.item(), ' (', prediction_score.squeeze().item(), ')')#print label

#위의 일반적인 과정을 AttirbutionVisualizer을 이용하여 시각화 할건데 인수는 모델의 배열, topk예측추출하기위한 스코어링, 훈련된클래스 목록, 찾아야하는 기능(ImageFeature), 데이터세트이다
from captum.insights. import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

def baseline_func(input):#all-zero input
    return input*0

def full_img_transform(input):#merge transforming
    i=Image.open(input)#open
    i=transform(i)#transform
    i=transform_normalize(i)#normalize
    i=i.unaqueeze(0)#extend
    return i
input_imgs=torch.cat(list(map(lambda i: full_img_transform(i), imgs)), 0)#concat full_img_transform, imgs

visualizer=AttributionVisualizer(#instantiation AttributionVisualizer
    models=[model],#our model as array
    score_func=lambda o: torch.nn.functional.softmax(o, 1),#scoring(점수화?) for extracting predict
    classes=list(map(lambda k: idx_to_labels[k][1], idx_to_labels.keys())),#trained models per labels
    features=[#function target
        ImageFeature(#we will find ImageFeature now
            "Photo",
            baseline_transforms=[baseline_func],
            input_transforms=[],
        )
    ],
    dataset=[Batch(input_imgs, labels=[282,849,69])]#dataset
)
visualizer.render()#Captum Insights위젯 렌더링. 속성과 인수 선택 후 예측된 클래스, 정확성을 기반으로 모델 응답을 필터링하고, 활률과 모델의 예측을 확인하고, 원래이미지와 비교하여 속성의 히트맵을 볼 수 있다.
