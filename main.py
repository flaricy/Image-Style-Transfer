import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter
# ---- some info below ----

# ! modify according to your device
device = torch.device("cuda")
    #  = d2l.try_gpu() 

pretrained_net = torchvision.models.vgg19(pretrained=True)

style_layers, content_layers = [0, 5, 10, 19, 28], [25]

# construct a new network instance net, which only retains all the VGG layers to be used for feature extraction.
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)]) # here "+" means list concatenation. "net" only contains 0~28-th layer of VGG


# ---- For preprocess and postprocess, implement image transformation ----

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# --- extract_features ----
def extract_features(X, content_layers, style_layers):
    """
        extract_features -- use global variable "net" to iteratively implement forward propogation of CNN, with respect to image X, 
                            while extract contents and styles according to content_layers and style_layers.
    """
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# The following 2 func can be invoked before training
def get_contents(image_shape, device):
    """
        get_contents --  extracts content features from the content image
    """
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    """
        get_styles -- extracts style features from the style image
    """
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

# ---- loss functions ----
# content_loss
def content_loss(Y_hat, Y):
    # We detach the target content from the tree used to dynamically compute
    # the gradient: this is a stated value, not a variable. Otherwise the loss
    # will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()

# style loss
def gram(X):
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

# total variation loss

# ? what the shape of Y_hat
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())

# ! hyper_param
content_weight, style_weight, tv_weight = 8, 1e4, 0

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

# ---- initialize the synthesized image ----

class SynthesizedImage(nn.Module):
    """treat our synthesized image as a net, whose only parameter is the image itself."""
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    """ initialize gen_img to our content image (i.e. X) """
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


# ---- training ----
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    # creat summary writer
    writer = SummaryWriter()

    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss',
    #                         xlim=[10, num_epochs],
    #                         legend=['content', 'style', 'TV'],
    #                         ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        print(f"epoch {epoch}")
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        
        l.backward()
        trainer.step()
        scheduler.step()
        # if (epoch + 1) % 10 == 0:
        #     animator.axes[1].imshow(postprocess(X))
        #     animator.add(epoch + 1, [float(sum(contents_l)),
        #                              float(sum(styles_l)), float(tv_l)])
            #d2l.plt.show()
        loss = l.item()
        writer.add_scalar("Loss/train", loss, epoch)
    return X


image_shape = (300, 450)  # PIL Image (h, w)
net = net.to(device)

# ---- content image and style image ----
name_content = "river"
name_style = "starry"

d2l.set_figsize()
content_img = d2l.Image.open(f'./images/{name_content}.jpeg').convert("RGB")
# no attribute: shape
# d2l.plt.imshow(content_img)
style_img = d2l.Image.open(f'./styles/{name_style}.jpeg')
#print("style image: ", style_img.shape)

# d2l.plt.imshow(style_img);
# d2l.plt.show()

content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.3, 2000, 200)

output_img = postprocess(output)
output_img.save(f'./results/{name_content + "-" + name_style}_random.jpeg', 'JPEG')
