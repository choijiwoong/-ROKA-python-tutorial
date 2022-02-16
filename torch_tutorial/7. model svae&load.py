"""
저장과 불러오기를 통해 모델의 상태를 persist하고 이를 사용할 수 있는데, pytorch는 학습한 매개변수를 state_dict라는 internal state dictionary에 저장한다.
저장과 불러오기는 torch.save, load_state_dict()메소드를 사용한다. inference전에 model.eval메소드를 호출하여 dropout과 batch nomalization을 evaluation mode로 설정해야한다.
만약 모델클래스 구조를 모델과 함께 저장하고 싶다면 mode.state_Dict()가 아닌 model을 전달해야한다. 이들은 pickle을 이용하여 serialize하기에 definition을 rely on해야한다.
"""
import torch
import torchvision.models as models#for save&load
# normal model
#save
model=models.vgg16(pretrained=True)#make random parameter model by models module
torch.save(model.state_dict(), 'model_wdight.pth')
#load
model=models.vgg16()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()#set model's dropout&batch normalization to evaluation mode

# user defined class with model
torch.save(model, 'model.pth')#not use model.state_dict()
model=torch.load('model.pth')#use just load
