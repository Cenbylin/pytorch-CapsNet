# coding:utf-8
import torch
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def has_nan(x):
    test1 = x != x
    return np.sum(test1) > 0


def _keep_dim_sum(T, sdims):
    """

    :param T:
    :type T: Variable
    :param sdims:
    :return:
    """
    T_ = T  # type: Variable
    if isinstance(sdims, int):
        T_ = T_.sum(dim=sdims, keepdim=True)
    else:
        for d in sdims:
            T_ = T_.sum(dim=d, keepdim=True)
    return T_


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        # Mnist 28*28
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
        self.pre_net = nn.Sequential(conv1, relu1, primarycaps)

        # RouteCap层 (6x6x32个8D-capsule路由到10个16D-capsule)
        self.routeCap1 = RouteCap(in_caps_num=6 * 6 * 32,
                                  in_caps_dim=8,
                                  out_caps_num=256,
                                  out_caps_dim=8)
        self.routeCap2 = RouteCap(in_caps_num=256,
                                  in_caps_dim=8,
                                  out_caps_num=400,
                                  out_caps_dim=3)
        # 重构原图
        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=3, out_channels=1,
                kernel_size=9, stride=1),
            nn.Tanh())
        # 损失
        self.loss = nn.MSELoss()

    def forward(self, input):
        c, a = self.pre_net(input)
        c, a = self.routeCap1(c, a)
        c, a = self.routeCap2(c, a)

        # c[batch, out_cap_num, out_cap_dim]展成2d
        c_2d = c.transpose(1, 2)
        # c_2d[batch, out_cap_dim, out_cap_num]
        shape_size = int(math.sqrt(c_2d.size(2)))
        c_2d = c_2d.contiguous().view(
            c_2d.size(0), c_2d.size(1),
            shape_size, shape_size)

        return self.encoder(c_2d)


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

        self.cells = [self.create_cell_fn(i) for i in range(out_caps_group)]
        self.a_conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_caps_group,
                                kernel_size=kernel_size,
                                stride=stride)
        self.a_relu = nn.ReLU(inplace=True)

    def forward(self, input):
        """
        group*h*w个out_dim-D capsule， 和group*h*w个activation
        :param input:[batch, in_channel, height, weight]
        :return: c[batch, out_caps_num, out_caps_dim] a[batch, out_caps_num]
        """
        # 得到group*[batch, out_caps_dim, out_h, out_w]
        _c = [self.cells[i](input) for i in range(self.out_caps_group)]  # 32
        # c[batch, out_caps_dim, group, out_h, out_w]
        c = torch.stack(_c, dim=2)
        # c[batch, out_caps_dim, group * out_h * out_w]
        c = c.view(c.size(0), c.size(1), -1)
        c = c.transpose(1, 2)

        a = self.a_relu(self.a_conv(input))
        a = a.view(a.size(0), -1)
        return c, a

    def create_cell_fn(self, group):
        """
        卷积处理，加入module
        :param group: 分组数
        :return:
        """
        conv1 = nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.out_caps_dim,
                          kernel_size=self.kernel_size,
                          stride=self.stride)
        name = "__capconv{}".format(group)
        # 确保会被记录parameter
        self.add_module(name, conv1)

        return conv1


class RouteCap(nn.Module):
    def __init__(self, in_caps_num, in_caps_dim,
                 out_caps_num, out_caps_dim):
        super(RouteCap, self).__init__()
        self.in_caps_num = in_caps_num
        self.in_caps_dim = in_caps_dim
        self.out_caps_num = out_caps_num
        self.out_caps_dim = out_caps_dim
        # batch共享一份权重
        self.W = nn.Parameter(torch.randn(out_caps_num, in_caps_num, in_caps_dim, out_caps_dim).type(FloatTensor))
        # 初始化联系矩阵模板
        # self.R = [Variable(
        #     torch.zeros(in_caps_num, out_caps_num).type(FloatTensor),
        #     requires_grad=False) + (1/out_caps_num)] * 10
        self.beta_v = 0.1
        self.lbda = 0.01
        self.beta_a = nn.Parameter(torch.randn(1, 1, 1, 1).type(FloatTensor), requires_grad=True)

        self.norm_const = - 0.5 * math.log(2.0 * 3.1416) * out_caps_dim

    def forward(self, c, a):
        """
        :param c: [batch, in_cap_num, in_cap_dim]
        :param a: [batch, in_cap_num]
        :return:c[batch, out_cap_num, out_cap_dim] a[batch, out_cap_num]
        """
        batch_size = c.size(0)

        # _c [batch, 1, in_cap_num, 1, in_cap_size]
        _c = c.unsqueeze(dim=1).unsqueeze(dim=3)

        # batch共享权重
        # W_batch [1, out_caps_num, in_caps_num, in_caps_dim, out_caps_dim]
        W_batch = self.W.unsqueeze(dim=0)

        # 矩阵相乘，得到下层对上层贡献
        # contribution: [batch, out_caps_num, in_cap_num, out_caps_dim]
        # [batch, 1, in_cap_num, 1, in_cap_dim] matmul [1, out_caps_num, in_caps_num, in_caps_dim, out_caps_dim]
        con = torch.matmul(_c, W_batch).squeeze()

        # 路由
        # 初始化联系R[batch, out_cap_num, in_cap_num]
        R = Variable(torch.zeros(batch_size,
                                 self.out_caps_num,
                                 self.in_caps_num).type(FloatTensor),
                     requires_grad=False) + (1 / self.out_caps_num)
        m, sigma, out_a = self.m_step(R, con, a)
        for _ in range(3):
            R = self.e_step(m, sigma, out_a, con)
            m, sigma, out_a = self.m_step(R, con, a)

        return m.squeeze(), out_a.squeeze()

    def e_step(self, m, sigma, out_a, con):
        """
        :param m:    [batch, out_caps_num, 1, out_caps_dim]
        :param sigma:[batch, out_caps_num, 1, out_caps_dim]
        :param out_a:[batch, out_caps_num, 1, 1]
        :param con:  [batch, out_caps_num, in_cap_num, out_caps_dim]
        :return: R[batch, out_cap_num, in_cap_num]
        """
        # nm_dv[batch, out_caps_num, in_cap_num, out_caps_dim]
        nm_dv = (con - m) ** 2
        nm_log1 = -0.5 * nm_dv / (sigma + 1e-6)
        # nm_log1[batch, out_caps_num, in_cap_num, 1]
        nm_log1 = _keep_dim_sum(nm_log1, sdims=3)

        # nm_log2[batch, out_caps_num, 1, 1]
        nm_log2 = -0.5 * _keep_dim_sum(torch.log(sigma), sdims=3) + self.norm_const

        # p[batch, out_caps_num, in_cap_num, 1]
        p_log = nm_log1 + nm_log2
        # 减少中间数值
        p_log_max = torch.max(p_log)
        p_log -= p_log_max
        p = torch.exp(p_log)

        # r[batch, out_caps_num, in_cap_num, 1] _r_sum[batch, 1, in_cap_num, 1]
        _r = p * out_a
        _r_sum = _keep_dim_sum(_r, sdims=1)

        r = _r / (_r_sum + 1e-6)
        return r.squeeze()

    def m_step(self, R, con, a):
        """
        :param R:[batch, out_cap_num, in_cap_num]
        :param con:[batch, out_caps_num, in_cap_num, out_caps_dim]
        :param a:[batch, in_cap_num]
        :return: m[batch, out_caps_num, 1, out_caps_dim]
             sigma[batch, out_caps_num, 1, out_caps_dim]
             out_a[batch, out_caps_num, 1, 1]
        """
        R = R.squeeze()
        con = con.squeeze()
        a = a.squeeze()

        # Ax[batch, 1, in_cap_num]
        Ax = a.unsqueeze(dim=1)
        # r[batch, out_cap_num, in_cap_num, 1]
        r = (R * Ax + 1e-6).unsqueeze(dim=3)
        # print("r", r.squeeze())
        # r_sum[batch, out_cap_num, 1, 1]
        r_sum = _keep_dim_sum(r, sdims=(2)) + 1e-6

        # m[batch, out_caps_num, 1, out_caps_dim]
        m = _keep_dim_sum(r * con, sdims=(2)) / r_sum

        # sigma[batch, out_caps_num, 1, out_caps_dim]
        sigma = _keep_dim_sum((con - m) ** 2, sdims=(2)) / r_sum

        # cost[batch, out_caps_num, 1, out_caps_dim]
        cost = (self.beta_v + torch.log(sigma) * 0.5) * r_sum
        # cost_sum[batch, out_caps_num, 1, 1]
        cost_sum = _keep_dim_sum(cost, sdims=(3))
        # print((self.lbda * (self.beta_a - cost_sum)).squeeze())
        out_a = torch.sigmoid(self.lbda * (self.beta_a - cost_sum))

        return m, sigma, out_a


if __name__ == '__main__1':
    from torch.optim import lr_scheduler
    from torchvision import datasets, transforms
    from torch.autograd import Variable
    import torch
    import torch.optim as optim
    # 图片显示
    import matplotlib.pyplot as plt
    import numpy as np


    def grayshow(img):
        img = img.squeeze()
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(npimg, cmap='gray')
        plt.show()


    batch_size = 16
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist', train=True, download=True, transform=data_transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist', train=False, download=True, transform=data_transform),
        batch_size=batch_size * 10, shuffle=True)

    net = CapsNet()
    net.cuda()
    print(net)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5)
    for epoch in range(30):
        # Update learning rate
        scheduler.step()
        print('Learning rate: {}'.format(scheduler.get_lr()[0]))
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data = Variable(data).cuda()
            # 跳过不够数量的批次
            if data.size() != torch.Size([batch_size, 1, 28, 28]):
                continue
            optimizer.zero_grad()
            output = net(data)
            loss = net.loss(output, data)
            loss.backward()
            optimizer.step()
            # adjust_learning_rate(optimizer, loss.data[0])
            if batch_idx % 16 == 0:
                print("loss", loss.data[0])
                if loss.data[0] < 0.07:
                    grayshow(output[0].cpu().data)
