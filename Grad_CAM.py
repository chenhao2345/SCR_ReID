# from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
# from torch import topk
import numpy as np
import skimage.transform
import torch
import os
# from model.resnet50_attention import AttnResNet50
# from prepare_attribute import AttributesDataset
# from model.attribute.resnet50 import ft_resnet50
# from model.identity.resnet50_id import ResNet50_id
# from model.attribute.vgg19 import ft_vgg19
from network import MGN
from data import Data
plt.ion()   # interactive mode

######################################################################
# Load Data and Model
# ---------

def tensor2image(tensor):
    inp = torch.squeeze(tensor)
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp=np.clip(inp,0,1)
    return inp


def load_network(network, name, which_epoch):
    save_path = os.path.join('./weights',name,'model_%s.pth'%which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if 'backbone' in name:
                x = module(x)
            if 'p1' == name:
                p1 = module(x)
                if name in self.target_layers:
                    # print(name)
                    # print(x.shape)
                    p1.register_hook(self.save_gradient)
                    outputs += [p1]
                    output = p1
            if 'p2' == name:
                p2 = module(x)
                if name in self.target_layers:
                    # print(name)
                    # print(x.shape)
                    p2.register_hook(self.save_gradient)
                    outputs += [p2]
                    output = p2
            if 'p3' == name:
                p3 = module(x)
                if name in self.target_layers:
                    # print(name)
                    # print(x.shape)
                    p3.register_hook(self.save_gradient)
                    outputs += [p3]
                    output = p3
            # if 'fc' not in name and 'pool' not in name:
            #     # print(name)
            #     # print(x.shape)
            #     x = module(x)
            #     # print(x.shape)
            #     if name in self.target_layers:
            #         # print(name)
            #         # print(x.shape)
            #         x.register_hook(self.save_gradient)
            #         outputs += [x]
                # print(outputs[0].shape)
        activations = outputs

        return activations, output


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        # self.feature_extractor = FeatureExtractor(self.model.features, target_layers)
        # ResNet object has no attribute 'features'
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x ,target):
        target_activations, output = self.feature_extractor(x)
        c0 = self.model.maxpool_zg_p2(output[:, :1024, :, :])
        c1 = self.model.maxpool_zg_p2(output[:, 1024:, :, :])
        whole = self.model.maxpool_zp2(output)
        s0 = whole[:, :, 0:1, :]
        s1 = whole[:, :, 1:2, :]
        s0 = self.model.reduction4(s0).squeeze(dim=3).squeeze(dim=2)
        s1 = self.model.reduction5(s1).squeeze(dim=3).squeeze(dim=2)
        s0 = self.model.fc_id_256_1_0(s0)
        s1 = self.model.fc_id_256_1_1(s1)
        c0 = self.model.reduction1_c0(c0).squeeze(dim=3).squeeze(dim=2)
        c1 = self.model.reduction1_c1(c1).squeeze(dim=3).squeeze(dim=2)
        c0 = self.model.fc_id_2048_0_c0(c0)
        c1 = self.model.fc_id_2048_0_c1(c1)

        if target=='c0':
            output = c0
        if target == 'c1':
            output = c1
        if target=='s0':
            output = s0
        if target == 's1':
            output = s1

        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, target,  index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda(),target)
        else:
            features, output = self.extractor(input,target)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        # print(index)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        # print(one_hot)
        self.model.zero_grad()
        # self.model.classifier_2_1.zero_grad()
        # self.model.classifier_2_2.zero_grad()
        # self.model.classifier_2_2.zero_grad()
        # self.model.classifier_2_2.zero_grad()
        # self.model.classifier_2_2.zero_grad()
        # self.model.classifier_2_2.zero_grad()
        # one_hot.backward(retain_variables=True)
        one_hot.backward()
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        # print(grads_val)
        # print(features.shape)
        target = features[-1]
        # print(target.shape)
        target = target.cpu().data.numpy()[0, :]
        # print(target.shape)
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        # print(weights.shape)
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = skimage.transform.resize(cam, (384,192))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model_structure = ft_resnet50()
# model = load_network(model_structure,'ft-resnet50','last')
# model_structure = ResNet50_id(751)
model_structure = MGN()
model = load_network(model_structure,'MGN-c2c3','500')
# model = load_network(model_structure,'ft_ResNet50','last')
model = model.to(device)
model.eval()

data = Data()
testset = data.testset
# print(testset._id2label[260])
for i in [4362]:
    input = testset[i][0]  # 4321
    print(testset[i][1])
    print(testset.imgs[i])
    # counter=0
    # for input,label in testset:
        # if label==82:
        #     print(counter)
        # counter+=1

    # print(input)
    input.unsqueeze_(0)
    input = input.to(device)

    grad_cam = GradCam(model=model, \
                           target_layer_names=["p2"], use_cuda=True)
    target_index = None
    # class_names = {'gender': {0:'male',1:'female'},
    #                'hair': {0:'short hair',1:'long hair'},
    #                'up': {0:'long sleeve', 1:'short sleeve'},
    #                'down': {0:'long lower body clothing', 1:'short lower body clothing'},
    #                'clothes': {0:'dress', 1:'pants'},
    #                'hat': {0:'no', 1:'yes'},
    #                'backpack':{0:'no', 1:'yes'},
    #                'bag':{0:'no', 1:'yes'},
    #                'handbag':{0:'no', 1:'yes'},
    #                'age': {0: 'young', 1: 'teenager', 2: 'adult', 3: 'old'},
    #                'upcolor': {0: 'black', 1: 'white', 2: 'red', 3: 'purple', 4: 'yellow', 5: 'gray', 6: 'blue',
    #                            7: 'green'},
    #                'downcolor': {0: 'black', 1: 'white', 2: 'pink', 3: 'purple', 4: 'yellow', 5: 'gray', 6: 'blue',
    #                              7: 'green', 8: 'brown'}}
    # entire_mask = np.zeros((256,128))
    # for i in range(0,12):
    #     mask = grad_cam(input,i, target_index)
    #     entire_mask = np.maximum(mask,entire_mask)
    #     plt.subplot(4, 3, i+1)
    #     plt.imshow(tensor2image(input))
    #     plt.imshow(mask, alpha=0.3, cmap='jet')
        # names = list(class_names.keys())
        # plt.title(names[i])

    # plt.subplot(1, 3, 1)
    # plt.imshow(tensor2image(input))
    # plt.title('original image')
    # plt.subplot(1, 3, 2)
    # a= torch.mul(input,torch.from_numpy(entire_mask).float().cuda())
    # plt.imshow(tensor2image(a))
    # plt.title('soft masked image')
    # plt.subplot(1, 3, 3)
    # entire_mask[entire_mask<0.4]=0
    # entire_mask[entire_mask>=0.4]=1
    # a= torch.mul(input,torch.from_numpy(entire_mask).float().cuda())
    # plt.imshow(tensor2image(a))
    # plt.title('hard masked image')

    # plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.subplot(1, 5, 1)
    plt.axis('off')
    plt.imshow(tensor2image(input))
    plt.subplot(1, 5, 2)
    plt.axis('off')
    mask = grad_cam(input, 's0', target_index)
    plt.imshow(tensor2image(input))
    plt.imshow(skimage.transform.resize(mask, (384,192),order=1,mode='constant',anti_aliasing=True), alpha=0.3, cmap='jet')
    plt.subplot(1, 5, 3)
    plt.axis('off')
    mask = grad_cam(input, 's1', target_index)
    plt.imshow(tensor2image(input))
    plt.imshow(skimage.transform.resize(mask, (384,192),order=1,mode='constant',anti_aliasing=True), alpha=0.3, cmap='jet')
    plt.subplot(1, 5, 4)
    plt.axis('off')
    mask = grad_cam(input, 'c0', target_index)
    plt.imshow(tensor2image(input))
    plt.imshow(skimage.transform.resize(mask, (384,192),order=1,mode='constant',anti_aliasing=True), alpha=0.3, cmap='jet')
    plt.subplot(1, 5, 5)
    plt.axis('off')
    mask = grad_cam(input, 'c1', target_index)
    plt.imshow(tensor2image(input))
    plt.imshow(skimage.transform.resize(mask, (384,192),order=1,mode='constant',anti_aliasing=True), alpha=0.3, cmap='jet')
    plt.title(i)
    plt.pause(100)