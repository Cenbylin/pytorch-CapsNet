# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import os

# 目前最好30，rl：0.0015
batch_size = 30

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', train=True, download=True, transform=data_transform),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', train=False, download=True, transform=data_transform),
                                          batch_size=batch_size, shuffle=True)


class CapsNet(nn.Module):
    global batch_size

    def __init__(self):
        super(CapsNet, self).__init__()
        self.build()

    def build(self):
        # 第一层卷积（1：灰度图通道 256: 第一层卷积通道 9: 9x9卷积核 1: 步长1）
        conv1 = nn.Conv2d(in_channels=1,
                          out_channels=256,
                          kernel_size=9,
                          stride=1)
        relu1 = nn.ReLU(inplace=True)
        # primarycaps层
        primarycaps = PrimaryCaps(in_channels=256,
                                  kernel_size=9,
                                  stride=2,
                                  out_caps_group=32,
                                  out_caps_dim=8)
        # route层 (6x6x32个8D-capsule路由到10个16D-capsule)
        route1 = Route(in_caps_num=6 * 6 * 32,
                       in_caps_dim=8,
                       out_caps_num=10,
                       out_caps_dim=16,
                       batch_size=batch_size)
        # 重构解码
        self.Decoder = Decoder()

        # Use GPU if available
        if torch.cuda.is_available():
            conv1 = conv1.cuda()
            primarycaps = primarycaps.cuda()
            route1 = route1.cuda()
            self.Decoder = self.Decoder.cuda()

        # 构成capsule网络
        self.net = nn.Sequential(conv1, relu1, primarycaps, route1)

    def forward(self, input):
        return self.net(input)

    def marginal_loss(self, v, target, l=0.5):
        # v: [batch_size, 10, 16]
        # target`: [batch_size, 10]
        # l: Scalar, lambda for down-weighing the loss for absent digit classes
        # L_c = T_c * max(0, m_plus - norm(v_c)) ^ 2 + lambda * (1 - T_c) * max(0, norm(v_c) - m_minus) ^2
        batch_size = v.size(0)
        square = v ** 2
        square_sum = torch.sum(square, dim=2)
        # norm: [batch_size, 10]
        norm = torch.sqrt(square_sum)
        assert norm.size() == torch.Size([batch_size, 10])

        # The two T_c in Eq.4
        T_c = target.type(torch.FloatTensor)
        zeros = Variable(torch.zeros(norm.size()))
        # Use GPU if available
        if torch.cuda.is_available():
            zeros = zeros.cuda()
            T_c = T_c.cuda()
        # Eq.4
        marginal_loss = T_c * (torch.max(zeros, 0.9 - norm) ** 2) + \
                        (1 - T_c) * l * (torch.max(zeros, norm - 0.1) ** 2)
        marginal_loss = torch.sum(marginal_loss)

        return marginal_loss

    def reconstruction_loss(self, reconstruction, image):
        # reconstruction: [batch_size, 784] Decoder outputs of images
        # image: [batch_size, 1, 28, 28] MNIST samples
        batch_size = image.size(0)
        # image: [batch_size, 784]
        image = image.view(batch_size, -1)
        assert image.size() == (batch_size, 784)

        # Scalar Variable
        reconstruction_loss = torch.sum((reconstruction - image) ** 2)
        return reconstruction_loss

    def loss(self, v, target, image):
        # 最终损失 marginal_loss + 0.0005*reconstruction_loss
        batch_size = image.size(0)

        marginal_loss = self.marginal_loss(v, target)
        # Get reconstructions from the decoder network
        reconstruction = self.Decoder(v, target)
        reconstruction_loss = self.reconstruction_loss(reconstruction, image)

        # Scalar Variable
        loss = (marginal_loss + 0.0005 * reconstruction_loss) / batch_size

        return loss, marginal_loss / batch_size, reconstruction_loss / batch_size


class PrimaryCaps(nn.Module):
    """
    从第一层卷积再作一层，然后形成capsule分组
    """

    def __init__(self, in_channels, kernel_size, stride, out_caps_group, out_caps_dim):
        super(PrimaryCaps, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_caps_group = out_caps_group
        self.out_caps_dim = out_caps_dim

        # 应该是32
        self.cells = [self.create_cell_fn() for i in range(out_caps_group)]

    def forward(self, input):
        # 输入为[batch, 256, 20, 20]
        # 输出[batch, 1152, 8]

        # 模拟32组(每组6x6)个capsule
        u = [self.cells[i](input) for i in range(self.out_caps_group)]  # 32

        # list，多出一维。
        # u:[batch, 8, 6, 6] -> [batch, 8, 32, 6, 6]
        u = torch.stack(u, dim=2)

        # 方便作所有向量的长度计算，后续squash使用
        # u: [batch, 8, 32, 6, 6] -> [batch, 8, 1152]
        u = u.view(u.size(0), u.size(1), -1)
        # u: [batch, 8, 1152] -> [batch, 1152, 8]
        u = u.transpose(1, 2)

        # print "caps_out:", u
        # squash output，压缩
        return self.squash(u)

    def squash(self, input):
        # 论文公式v_j = (norm(s_j) ^ 2 / (1 + norm(s_j) ^ 2)) * (s_j / norm(s_j))
        # 输入 [batch, 1152, 8]
        # 输出 [batch, 1152, 8]

        # 计算向量长度的平方
        # mod_sq -> [batch, 1152]
        mod_sq = torch.sum(input ** 2, dim=2)
        # 向量长度
        mod = torch.sqrt(mod_sq)

        # factor for u: [batch, 1152] then factor [batch, 1152]
        factor = mod_sq / (mod * (1 + mod_sq))

        # u_squashed: [batch_size, 1152, 8]
        #            = [batch_size, 1152, 1] * [batch, 1152, 8]
        u_squashed = factor.unsqueeze(2) * input

        return u_squashed

    def create_cell_fn(self):
        """
        create sub-network inside a capsule.
        :return:
        """
        conv1 = nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.out_caps_dim,
                          kernel_size=self.kernel_size,
                          stride=self.stride)
        # Use GPU if available
        if torch.cuda.is_available():
            conv1 = conv1.cuda()

        return conv1


class Route(nn.Module):
    def __init__(self, in_caps_num, in_caps_dim,
                 out_caps_num, out_caps_dim, batch_size):
        super(Route, self).__init__()
        self.in_caps_num = in_caps_num
        self.in_caps_dim = in_caps_dim
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        self.batch_size = batch_size
        # batch共享一份权重
        # [1152, 10, 16, 8]
        self.W = nn.Parameter(torch.randn(in_caps_num, out_caps_num, out_caps_dim, in_caps_dim))

    def squash(self, input):
        # 论文公式v_j = (norm(s_j) ^ 2 / (1 + norm(s_j) ^ 2)) * (s_j / norm(s_j))
        # 输入 [batch, 10, 16]
        # 输出 [batch, 1152, 8]

        # 计算向量长度的平方
        # mod_sq -> [batch, 1152]
        mod_sq = torch.sum(input ** 2, dim=2)
        # 向量长度
        mod = torch.sqrt(mod_sq)

        # factor for u: [batch, 1152] then factor [batch, 1152]
        factor = mod_sq / (mod * (1 + mod_sq))

        # u_squashed: [batch_size, 1152, 8]
        #            = [batch_size, 1152, 1] * [batch, 1152, 8]
        u_squashed = factor.unsqueeze(2) * input

        return u_squashed

    def softmax(self, input_, dim):
        # input_ex = torch.exp(input)
        # return input_ex / input_ex.sum(dim, keepdim=True)
        batch_size = input_.size()[0]
        output_ = torch.stack([F.softmax(input_[i]) for i in range(batch_size)], 0)
        return output_

    def forward(self, input):
        # print "route input:", input
        # 输入[batch, 1152, 8]
        # 输出[batch, 10, 16]

        # 构造8x1的输入端
        # u: [batch, 1152, 8] -> [batch, 1152, 8, 1]
        u = torch.unsqueeze(input, dim=3)

        # 跟10个输出capsule都有运算
        # u_stack -> [batch_size, 1152, 10, 8, 1]
        u_stack = torch.stack([u for i in range(self.out_caps_num)], dim=2)

        # 批次共享权重
        # W_batch -> [batch_size, 1152, 10, 16, 8]
        W_batch = torch.stack([self.W for i in range(self.batch_size)], dim=0)

        # 矩阵相乘
        # u_hat: [batch_size, 1152, 10, 16]
        # [100, 1152, 10, 16, 8] [batch_size, 1152, 10, 8, 1]
        # print "W_batch:", W_batch.size(), "u_stack:", u_stack.size()
        u_hat = torch.matmul(W_batch, u_stack).squeeze()
        # print "uhat:", u_hat

        # 初始化先验概率b_ij
        # [1152, 10]
        b = Variable(torch.zeros(self.in_caps_num, self.out_caps_num))
        # Use GPU if available
        if torch.cuda.is_available():
            b = b.cuda()
        # print "route-b:", b

        # start routing now.
        for _ in range(3):
            # c -> [1152, 10]
            # c = F.softmax(b, dim=1)
            c = self.softmax(b, dim=1)
            # 增加维数，适配下面的相乘
            # c -> [1, 1152, 10, 1]
            c = c.unsqueeze(2).unsqueeze(0)

            # 累加1152个capsule的贡献到10个中
            # u_hat: [batch_size, 1152, 10, 16]
            # u_hat * c -> [batch_size, 1152, 10, 16]
            # s -> [batch_size, 10, 16]
            # print "route_c:", c
            s = torch.sum(u_hat * c, dim=1)
            # print "route_s:", s

            # v -> [batch_size, 10, 16]
            v = self.squash(s)

            # b_ij += u_hat * v_j
            # u_hat: [batch_size, 1152, 10, 16]
            # v: [batch_size, 10, 16]
            # a: [batch_size, 10, 1152, 1]
            a = torch.matmul(u_hat.transpose(1, 2), v.unsqueeze(3))
            # b: [1152, 10]
            b = b + torch.sum(a.squeeze().transpose(1, 2), dim=0)

        # print "route_out:", v
        return v


class Decoder(nn.Module):
    def __init__(self):
        '''
        The decoder network consists of 3 fully connected layers, with
        512, 1024, 784 neurons each.
        '''
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(10 * 16, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)

    def forward(self, v, target):
        batch_size = target.size(0)

        target = target.type(torch.FloatTensor)
        # mask: [batch_size, 10, 16]
        # 仅仅目标的向量为1，其他为0
        mask = torch.stack([target for i in range(16)], dim=2)
        if torch.cuda.is_available():
            mask = mask.cuda()

        # v: [bath_size, 10, 16]
        # 相乘就只保留了激活向量
        v_masked = mask * v
        v_masked = v_masked.view(batch_size, -1)
        assert v_masked.size() == torch.Size([batch_size, 160])

        # Forward
        v = self.fc1(v_masked)
        v = self.fc2(v)
        reconstruction = F.sigmoid(self.fc3(v))

        return reconstruction


def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot


def test(model):
    test_batch_num = 10
    correct = 0

    # 测试方案：仅仅取一批数据，测试计算正确率
    for batch_idx, (data, target) in enumerate(test_loader):
        # Store the indices for calculating accuracy
        label = target.unsqueeze(0).type(torch.LongTensor)

        batch_size = data.size(0)
        # Transform to one-hot indices: [batch_size, 10]
        target = to_one_hot(target, 10)
        assert target.size() == torch.Size([batch_size, 10])

        data, target = Variable(data, volatile=True), Variable(target)
        # Use GPU if available
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        # Output predictions
        output = model(data)

        # Count correct numbers
        # norms: [batch_size, 10]
        # 计算向量长
        norms = torch.sqrt(torch.sum(output ** 2, dim=2))

        # pred: [batch_size]
        # 取最大,[1]取到了位置（30x1的格式）
        pred = norms.data.max(1, keepdim=True)[1].type(torch.LongTensor)
        # print "labelsize", label.view_as(pred)
        # view_as后是30x1
        # 得到一个batch中正确的数量
        correct += pred.eq(label.view_as(pred)).cpu().sum()
        if batch_idx == (test_batch_num - 1):
            break
    return (correct + 0.0) / (batch_size * test_batch_num)


def adjust_learning_rate(optimizer, loss):
    '''
    调整学习率
    '''
    global step
    if step == 0:
        if loss < 2:
            step += 1
            print "【adjust rl to", optimizer.param_groups[0]['lr'] * 0.1, "】"
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
    elif step == 1:
        if loss < 1:
            step += 1
            print "【adjust rl to", optimizer.param_groups[0]['lr'] * 0.1, "】"
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
    elif step == 2:
        if loss < 0.6:
            step += 1
            print "【adjust rl to", optimizer.param_groups[0]['lr'] * 0.1, "】"
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.03
    elif step == 3:
        if loss < 0.3:
            step += 1
            print "【adjust rl to", optimizer.param_groups[0]['lr'] * 0.07, "】"
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.07
    else:
        pass


if __name__ == '__main__1':
    clip = 5
    net = CapsNet()
    net.cuda()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=0.0015)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5)
    need_adj = True
    for epoch in range(30):
        # Update learning rate
        scheduler.step()
        print('Learning rate: {}'.format(scheduler.get_lr()[0]))
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            target_onehot = to_one_hot(target, 10)
            data, target = Variable(data).cuda(), Variable(target_onehot).cuda()
            # assert data.size() == torch.Size([batch_size, 1, 28, 28])
            # 跳过不够数量的批次
            if data.size() != torch.Size([batch_size, 1, 28, 28]):
                continue
            optimizer.zero_grad()
            output = net(data)
            loss, m_loss, r_loss = net.loss(output, target, data)
            loss.backward()
            # 防止梯度爆炸
            # torch.nn.utils.clip_grad_norm(net.parameters(), clip)
            optimizer.step()
            if batch_idx % 20 == 0 and batch_idx != 0:
                # print "!!!out put:", output
                print "loss", loss.data[0], "r_loss", r_loss.data[0]
                # adjust_learning_rate(optimizer, loss.data[0])
            if batch_idx % 50 == 0 and batch_idx != 0 and loss.data[0] < 1.0:
                print "test begin"
                correct_rate = test(net)
                print "correct_rate:", correct_rate
                # 保存网络
                # if correct_rate > 0.98:
                #    torch.save(net, 'caps-net.pkl')
