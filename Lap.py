import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
# ---- some info below ----

# modify according to your device
DEVICE = d2l.try_gpu()

pretrained_net = torchvision.models.vgg19(pretrained=True)

Style_Layers, Content_Layers = [0, 5, 10, 19, 28], [21]

# construct a new network instance net, which only retains all the VGG layers to be used for feature extraction.
Incomplete_CNN = nn.Sequential(*[pretrained_net.features[i] for i in
                                 range(max(Content_Layers + Style_Layers) + 1)])
# here "+" means list concatenation. "net" only contains 0~28-th layer of VGG


# ---- For preprocess and postprocess, implement image transformation ----

RGB_MEAN = torch.tensor([0.485, 0.456, 0.406])
RGB_STD = torch.tensor([0.229, 0.224, 0.225])


def preprocess(img, image_shape_, blur=False):
    """
    预处理图片
    :param img: 图片
    :param image_shape_: resize的图片大小
    :return: 预处理过后的图片，并加入一个空维
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape_),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)])
    if blur:
        transforms = torchvision.transforms.Compose([
            transforms,
            torchvision.transforms.GaussianBlur(kernel_size=3)
        ])
    return transforms(img).unsqueeze(0)


def postprocess(img):
    """
    将img转为PILImage格式
    """
    img = img[0].to(RGB_STD.device)
    img = torchvision.transforms.Resize(Content_Size)(img)
    img = torch.clamp(img.permute(1, 2, 0) * RGB_STD + RGB_MEAN, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


# --- extract_features ----
def extract_features(X, content_layers_, style_layers_):
    """
    use global variable "net" to iteratively implement forward propogation of CNN, with respect to image X, while extract contents and styles according to content_layers and style_layers.
    :param content_layers_: content层的编号
    :param style_layers_: style层的编号
    :return: 抽取出来的content层和style层
    """
    contents = []
    styles = []
    for i in range(len(Incomplete_CNN)):
        X = Incomplete_CNN[i](X)
        if i in style_layers_:
            styles.append(X)
        if i in content_layers_:
            contents.append(X)
    return contents, styles


# The following 2 func can be invoked before training
def get_contents(image_shape_, device_):
    """
    extracts content features from the content image
    :param image_shape_: 预处理resize图像的大小
    :param device_: gpu/cpu
    :return: (预处理过后的content图片, 从content图片中抽取出来的content层, content的拉普拉斯)
    """

    content_X_ = preprocess(Content_Img, image_shape_).to(device_)
    content_X_blur = preprocess(Content_Img, image_shape_, blur=True).to(device_)
    contents_Y_, _ = extract_features(content_X_, Content_Layers, Style_Layers)
    lap_Y_ = lap_conv(content_X_blur)
    return content_X_, contents_Y_, lap_Y_


def get_styles(image_shape_, device_):
    """
    extracts style features from the style image
    :param image_shape_: 预处理resize图像的大小
    :param device_: gpu/cpu
    :return: (预处理过后的style图片, 从style图片中抽取出来的style层)
    """
    style_X = preprocess(Style_Img, image_shape_).to(device_)
    _, styles_Y_ = extract_features(style_X, Content_Layers, Style_Layers)
    return style_X, styles_Y_


# ---- loss functions ----
# content_loss
def content_loss(Y_hat, Y):
    """
    求content loss
    We detach the target content from the tree used to dynamically compute the gradient: this is a stated value, not a variable. Otherwise the loss will throw an error.
    :param Y_hat: content层的预测值
    :param Y: content层的参考值（从content图中抽取出来的content层）
    :return: content层的loss
    """
    return torch.square(Y_hat - Y.detach()).mean()


# style loss
def gram(X):
    """
    将张量X reshape成形状为(num_channels, -1)的矩阵T，返回T * T转置
    """
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / n


def style_loss(Y_hat, gram_Y):
    """
    style loss
    :param Y_hat: style层的预测值
    :param gram_Y: 从style图中抽取出来的style层的gram矩阵
    :return: style loss
    """
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean() / 4

# ? what the shape of Y_hat
def tv_loss(Y_hat):
    """
    total variation loss
    """
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


def lap_conv(X):
    """
    每个通道分别进行拉普拉斯卷积，最后将绝对值相加
    :param X: 被卷积的张量，四维NCHW格式
    :param device_: cpu/gpu
    :return:
    """
    X_ = nn.functional.avg_pool2d(X, kernel_size=5, padding=2, stride=1) # 3维张量
    kernel = [[0, -1, 0], [-1, 4, -1], [0, -1, 0]] # 拉普拉斯卷积核
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0) # 4维张量
    weight = nn.Parameter(data=kernel, requires_grad=False)
    weight = weight.to(DEVICE)
    X1 = X_[:, 0]
    X2 = X_[:, 1]
    X3 = X_[:, 2]
    x1 = nn.functional.conv2d(X1.unsqueeze(1), weight, padding=1)
    x2 = nn.functional.conv2d(X2.unsqueeze(1), weight, padding=1)
    x3 = nn.functional.conv2d(X3.unsqueeze(1), weight, padding=1)
    x1 = torch.abs(x1)
    x2 = torch.abs(x2)
    x3 = torch.abs(x3)
    rtn = x1 + x2 + x3
    return rtn

def lap_loss(lap_Y, lap_Y_hat):
    """
    拉普拉斯损失
    :param lap_Y: 拉普拉斯的参考值
    :param lap_Y_hat: 拉普拉斯的预测值
    :return: 拉普拉斯损失
    """
    return torch.sum(torch.square(lap_Y - lap_Y_hat))

# ! hyper_param
content_weight, style_weight, tv_weight, lap_weight = 8, 3500, 0, 0


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y_, styles_Y_gram, lap_Y_, lap_Y_hat):
    """
    Calculate the content, style, and total variance losses respectively
    :param X: 当前生成的图片
    :param contents_Y_hat: 内容层的预测
    :param styles_Y_hat: 风格层的预测
    :param contents_Y_: 内容层的参考值
    :param styles_Y_gram: 风格层gram的参考值
    :param lap_Y_: 拉普拉斯的参考值
    :param lap_Y_hat: 拉普拉斯的预测值
    :return: (内容loss, 风格loss, 方差loss, 拉普拉斯loss, 总loss)
    """
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y_)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    lap_l = lap_loss(lap_Y_, lap_Y_hat) * lap_weight
    # Add up all the losses
    l_ = sum(styles_l) / 5 + sum(contents_l) + sum([tv_l]) + sum([lap_l])
    return contents_l, styles_l, tv_l, lap_l, l_


# ---- initialize the synthesized image ----

class SynthesizedImage(nn.Module):
    """
    treat our synthesized image as a net, whose only parameter is the image itself.
    """
    def __init__(self, img_shape, **kwargs):
        """
        :param img_shape: 预处理resize的图片大小
        :param kwargs: 其他需要初始化父类的参数
        """
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(preprocess(Style_Img, IMAGE_SHAPE))
        # self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        """
        forward的过程就是直接返回weight
        :return: self.weight
        """
        return self.weight


def get_inits(X, device_, lr, styles_Y_):
    """
    initialize gen_img to our content image (i.e. X)
    :param X: 初始的图像
    :param device_: gpu/cpu
    :param lr: 学习率
    :param styles_Y_: 风格层的参考值
    :return: (第一次预测的图片, 风格层gram参考值, 训练器)
    """
    gen_img = SynthesizedImage(X.shape).to(device_)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.LBFGS(gen_img.parameters(), lr=lr)
    # trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y_]
    return gen_img(), styles_Y_gram, trainer


# ---- training ----
def train(X, contents_Y_, styles_Y_, device_, lr, num_epochs, lr_decay_epoch):
    """
    训练
    :param X: 初始图像
    :param contents_Y_: 图像层参考值
    :param styles_Y_: 风格层参考值
    :param device_: gpu/cpu
    :param lr: 学习率
    :param num_epochs: epochs的个数
    :param lr_decay_epoch: 间隔多少个epoch降低一下学习率，用于StepLR的参数
    :return: 经过num_epochs次迭代后，最终生成的图像
    """
    writer = SummaryWriter()

    X, styles_Y_gram, trainer = get_inits(X, device_, lr, styles_Y_)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    for epoch in range(num_epochs):
        print(f"epoch {epoch}:", end='\t')
        trainer.zero_grad()  # 清空当前梯度
        if isinstance(trainer, torch.optim.LBFGS):  # 训练器采用L-BFGS
            def closure():
                trainer.zero_grad()
                contents_Y_hat, styles_Y_hat = extract_features(
                    X, Content_Layers, Style_Layers)
                lap_Y_hat = lap_conv(X)
                _, _, _, _, ll = compute_loss(
                    X, contents_Y_hat, styles_Y_hat, contents_Y_, styles_Y_gram, Lap_Y, lap_Y_hat)
                ll.backward()
                loss_ = ll.item()
                return ll

            trainer.step(closure)

            contents_Y_hat, styles_Y_hat = extract_features(
                X, Content_Layers, Style_Layers)
            lap_Y_hat = lap_conv(X)
            _, _, _, _, ll = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y_, styles_Y_gram, Lap_Y, lap_Y_hat)
            loss = ll.item()
            print(f"Loss = {loss}")
            writer.add_scalar("Loss/train", loss, epoch)

        else:
            contents_Y_hat, styles_Y_hat = extract_features(
                X, Content_Layers, Style_Layers
            )
            lap_Y_hat = lap_conv(X)
            _, _, _, _, ll = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y_, styles_Y_gram, Lap_Y, lap_Y_hat
            )
            ll.backward()
            trainer.step()
            loss_ = ll.item()  # item：可以将一个零维的张量转成int或者float等类型
            writer.add_scalar("Loss/train", loss_, epoch)
            print(f"Loss = {loss_}")
        if (epoch + 1) % 20 == 0:
            img = X[0].to(RGB_STD.device)
            img = torchvision.transforms.Resize(Content_Size)(img)
            img = torch.clamp(img.permute(1, 2, 0) * RGB_STD + RGB_MEAN, 0, 1).permute(2, 0, 1)
            writer.add_image("Image", img, (epoch + 1) // 20)
        scheduler.step()
    return X


IMAGE_SHAPE = (400, 350)  # PIL Image (h, w)
Incomplete_CNN = Incomplete_CNN.to(DEVICE)

# ---- content image and style image ----
Content_Path = "./images/"
Content_Name = "Alps"
Style_Path = './styles/'
Style_Name = "Kandinsky"

d2l.set_figsize()
Content_Img = d2l.Image.open(Content_Path + Content_Name + '.jpeg').convert("RGB")
Content_Size = Content_Img.size[1], Content_Img.size[0]
Style_Img = d2l.Image.open(Style_Path + Style_Name + '.jpeg')

Content_X, Content_Y, Lap_Y = get_contents(IMAGE_SHAPE, DEVICE)
_, Style_Y = get_styles(IMAGE_SHAPE, DEVICE)

Output = train(Content_X, Content_Y, Style_Y, DEVICE, 0.8, 200, 25)

Output_Img = postprocess(Output)
Output_Img.save(f'./results/{Content_Name + "-" + Style_Name}_gatys.jpeg', 'JPEG')
