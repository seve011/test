import numpy as np
import time
import cv2
from PIL import Image

bias_bit = 13
gam_bit = 16
bet_bit = 16
layer22_dec_bit = 13

result_table = np.zeros([7, 3], int)
result_table[0] = [1, 2, 3]
for i in range(1, 7, 1):
    result_table[i] = result_table[0] * (i + 1)


def lemon_3bit_conv(feature, weight):
    feature_shape = np.shape(feature)
    [c, h, wi] = np.shape(weight)
    if feature_shape == (c, h, wi):
        re = np.zeros_like(result_table)
        for i in range(c):
            for j in range(h):
                for k in range(wi):
                    f = int(feature[i, j, k])
                    w = int(weight[i, j, k])
                    if (0 < w < 4) & (0 < f < 8):
                        re[f - 1, w - 1] += result_table[f - 1, w - 1]
                    if (-4 < w < 0) & (0 < f < 8):
                        re[f - 1, -w - 1] -= result_table[f - 1, -w - 1]
    return np.sum(re)


def HLSeparate(A):
    [c, h, w] = np.shape(A)
    B = np.zeros([2, c, h, w], int)
    for i in range(c):
        for j in range(h):
            for k in range(w):
                B[0, i, j, k] = int((A[i, j, k]) / 8)
                B[1, i, j, k] = (A[i, j, k]) % 8
    return B


def lemon_6bit_conv(feature, weight):
    if np.shape(feature) == np.shape(weight):
        featureHL = HLSeparate(feature)
        reH = lemon_3bit_conv(featureHL[0], weight)
        reL = lemon_3bit_conv(featureHL[1], weight)
        re = 8 * reH + reL
    return re


def lemon_9bit_conv(feature, weight):
    if np.shape(feature) == np.shape(weight):
        featureHL = HLSeparate(feature)
        reH = lemon_6bit_conv(featureHL[0], weight)
        reL = lemon_3bit_conv(featureHL[1], weight)
        re = 8 * reH + reL
    return re


def lemon_conv3d(fms, weights, activation_bit):
    [c, h, w] = fms.shape
    [_, k, _] = weights.shape
    outputs = np.zeros([h, w], np.float32)
    r = int((k - 1) / 2)
    # 定义边界填充0后的map
    padding_fm = np.zeros([c, h + k - 1, w + k - 1], np.float32)
    # 将输入在指定该区域赋值，即除了4个边界后，剩下的区域
    padding_fm[:, r:h + r, r:w + r] = fms
    # 对每个点为中心的区域遍历
    for i in range(r, h + r, 1):
        for j in range(r, w + r, 1):
            # 取出当前点为中心的k*k区域
            roi = padding_fm[:, i - r:i + r + 1, j - r:j + r + 1]
            # 计算当前点的卷积,对k*k个点点乘后求和
            ov = int(i - r)
            oh = int(j - r)
            # print(ov,oh)
            outputs[ov][oh] = np.sum(roi * weights)
            """
            if activation_bit<4:
                outputs[ov][oh]=lemon_3bit_conv(roi,weights)
            elif activation_bit<10:
                outputs[ov][oh]=lemon_9bit_conv(roi,weights)
            else:
                print('Activation_bit is larger than 9!')
            """
    return outputs


def lemon_conv3d_v2(fms, weights):
    [c, h, w] = fms.shape
    # [_,k,_] = weights.shape
    outputs = np.zeros([h, w], np.float32)
    fout = np.zeros_like(fms[0])
    # 对每个feature map遍历，从而对每个feature map进行卷积
    for i in range(c):
        # feature map==>[h,w]
        # f_map=fms[i]
        # kernel ==>[k,k]
        # we=weights[i]
        # rs = compute_conv(f_map,we)
        cv2.filter2D(fms[i], -1, weights[i], fout, borderType=cv2.BORDER_CONSTANT)
        # print(np.max(f_map),np.max(we),np.max(fout))
        outputs = outputs + fout
    return outputs


fmap = np.array(
    [[[1.0, 2, 3, 4, 5], [4, 5, 6, 7, 8], [7, 9, 9, 10, 6]], [[3, 2, 3, 4, 5], [4, 5, 6, 7, 8], [5, 4, 3, 2, 6]]])
weightt = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
fmap1 = fmap.transpose(1, 2, 0)
print(np.shape(fmap1))
fout = np.zeros([3, 5, 2])
print(np.shape(fout))
cv2.filter2D(fmap1, -1, weightt, fout, borderType=cv2.BORDER_CONSTANT)
fout1 = fout.transpose(2, 0, 1)
print(fmap)
print(fout1)


def lemon_conv(I_arr, wei, activation_bit):
    [c, h, w] = I_arr.shape
    [kn, _, _, _] = wei.shape
    out = np.zeros([kn, h, w], np.float32)
    for i in range(kn):
        out[i] = lemon_conv3d(I_arr, wei[i], activation_bit)
        # print(i)
    return out


def lemon_conv_v2(I_arr, wei):
    [c, h, w] = I_arr.shape
    [kn, _, _, _] = wei.shape
    out = np.zeros([kn, h, w], np.float32)
    for i in range(kn):
        out[i] = lemon_conv3d_v2(I_arr, wei[i])
        # print(i)
    return out


def lemon_batch(I_arrs, ws, activation_bit):
    [batch_size, c, h, w] = I_arrs.shape
    [kn, _, _, _] = ws.shape
    outs = np.zeros([batch_size, kn, h, w], np.float32)
    for i in range(batch_size):
        outs[i] = lemon_conv(I_arrs[i], ws, activation_bit)
        # print(i)
    return outs


def lemon_batch_v2(I_arrs, ws):
    [batch_size, c, h, w] = I_arrs.shape
    [kn, _, _, _] = ws.shape
    outs = np.zeros([batch_size, kn, h, w], np.float32)
    for i in range(batch_size):
        outs[i] = lemon_conv_v2(I_arrs[i], ws)
        # print(i)
    return outs


def batch_norm(X, gamma, beta, mean, var, eps):
    if len(X.shape) == 4:
        X_hat = (X - mean) / np.sqrt(var + eps)
        Y = gamma * X_hat + beta
    else:
        print("Error")
    return Y


def leaky1(X):
    [batch_size, c, h, w] = np.shape(X)
    XX = np.zeros([2, batch_size, c, h, w])
    XX[0] = X
    XX[1] = np.zeros_like(X)
    # XX[1]=0.1*X
    Y = np.max(XX, 0)
    return (Y)


def CB0(activation, activation_bit, quan_w, gam, bias):
    quan_a = np.float32(activation)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    quan_conv = lemon_batch_v2(quan_a, quan_w)
    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_conv + bias) * gam

    # bn_out=leaky1(bn_out)
    return bn_out


def CM0(activation, activation_bit, quan_w, gam, bias):
    quan_a = np.float32(activation)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    quan_conv = lemon_batch_v2(quan_a, quan_w)
    #np.save('./lemon_data/layer0_conv.npy', quan_conv[0])

    quan_max = maxpool2d(quan_conv)
    #np.save('lemon_data/layer0_conv_max.npy', quan_max[0])

    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_max + bias) * gam
    #np.save('lemon_data/layer0_conv_max_add_mul.npy', 2 ** (gam_bit - 3) * bn_out[0])

    # bn_out=leaky1(bn_out)
    return bn_out


def CB(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    quan_conv = lemon_batch_v2(quan_a, quan_w)

    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_conv + bias) * gam

    # bn_out=leaky1(bn_out)
    return bn_out


def CM8(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    #np.save('lemon_data/layer8_quan.npy', quan_a[0])
    quan_conv = lemon_batch_v2(quan_a, quan_w)
    #np.save('lemon_data/layer8_quan_conv.npy', quan_conv[0])

    quan_max = maxpool2d(quan_conv)
    #np.save('lemon_data/layer8_quan_conv_max.npy', quan_max[0])
    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_max + bias) * gam
    #np.save('lemon_data/layer8_quan_conv_max_add_mul.npy', 2 ** (gam_bit - 3) * bn_out[0])

    # bn_out=leaky1(bn_out)
    return bn_out


def CB10(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    #np.save('lemon_data/layer10_quan.npy', quan_a[0])
    quan_conv = lemon_batch_v2(quan_a, quan_w)
    #np.save('lemon_data/layer10_quan_conv.npy', quan_conv[0])

    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_conv + bias) * gam
    #np.save('lemon_data/layer10_quan_conv_add_mul.npy', 2 ** (gam_bit - 3) * bn_out[0])

    # bn_out=leaky1(bn_out)
    return bn_out


def CB21(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    #np.save('lemon_data/layer21_quan.npy', quan_a[0])
    quan_conv = lemon_batch_v2(quan_a, quan_w)
    #np.save('lemon_data/layer21_quan_conv.npy', quan_conv[0])

    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_conv + bias) * gam
    #np.save('lemon_data/layer21_quan_conv_add_mul.npy', 2 ** (gam_bit - 3) * bn_out[0])

    # bn_out=leaky1(bn_out)
    return bn_out


def CM(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    quan_conv = lemon_batch_v2(quan_a, quan_w)

    quan_max = maxpool2d(quan_conv)
    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_max + bias) * gam

    # bn_out=leaky1(bn_out)
    return bn_out


def CM2(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    #np.save('lemon_data/layer2_quan.npy', quan_a[0])
    quan_conv = lemon_batch_v2(quan_a, quan_w)
    #np.save('lemon_data/layer2_quan_conv.npy', quan_conv[0])

    quan_max = maxpool2d(quan_conv)
    #np.save('lemon_data/layer2_quan_conv_max.npy', quan_max[0])
    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_max + bias) * gam
    #np.save('lemon_data/layer2_quan_conv_max_add_mul.npy', 2 ** (gam_bit - 3) * bn_out[0])

    # bn_out=leaky1(bn_out)
    return bn_out


def CM4(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    #np.save('lemon_data/layer4_quan.npy', quan_a[0])
    quan_conv = lemon_batch_v2(quan_a, quan_w)
    #np.save('lemon_data/layer4_quan_conv.npy', quan_conv[0])

    quan_max = maxpool2d(quan_conv)
    #np.save('lemon_data/layer4_quan_conv_max.npy', quan_max[0])
    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_max + bias) * gam
    #np.save('lemon_data/layer4_quan_conv_max_add_mul.npy', 2 ** (gam_bit - 3) * bn_out[0])

    # bn_out=leaky1(bn_out)
    return bn_out


def CM6(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    # quan_a=np.uint8(quan_a)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    #np.save('lemon_data/layer6_quan.npy', quan_a[0])
    quan_conv = lemon_batch_v2(quan_a, quan_w)
    #np.save('lemon_data/layer6_quan_conv.npy', quan_conv[0])

    quan_max = maxpool2d(quan_conv)
    #np.save('lemon_data/layer6_quan_conv_max.npy', quan_max[0])
    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_max + bias) * gam
    #np.save('lemon_data/layer6_quan_conv_max_add_mul.npy', 2 ** (gam_bit - 3) * bn_out[0])

    # bn_out=leaky1(bn_out)
    return bn_out


def CB_v2(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    [c_out, c_in, kh, kw] = np.shape(quan_w)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    quan_conv = np.zeros([batch_size, c_out, h, wi])
    for i in range(c_out):
        quan_conv[:, i, :, :] = np.sum(quan_w[i:i + 1] * quan_a, 1)

    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_conv + bias) * gam
    # bn_out=leaky1(bn_out)
    return bn_out


def CB_v2_13(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    #np.save('lemon_data/layer13_quan.npy', quan_a[0])
    [c_out, c_in, kh, kw] = np.shape(quan_w)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    quan_conv = np.zeros([batch_size, c_out, h, wi])
    for i in range(c_out):
        quan_conv[:, i, :, :] = np.sum(quan_w[i:i + 1] * quan_a, 1)
    #np.save('lemon_data/layer13_quan_conv.npy', quan_conv[0])

    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_conv + bias) * gam
    #np.save('lemon_data/layer13_quan_conv_add_mul.npy', 2 ** (gam_bit - 3) * bn_out[0])
    # bn_out=leaky1(bn_out)
    return bn_out


def CB_v2_18(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    #np.save('lemon_data/layer18_quan.npy', quan_a[0])
    [c_out, c_in, kh, kw] = np.shape(quan_w)
    # quan_w=np.float64(weight)
    # print(np.shape(quan_w))
    quan_conv = np.zeros([batch_size, c_out, h, wi])
    for i in range(c_out):
        quan_conv[:, i, :, :] = np.sum(quan_w[i:i + 1] * quan_a, 1)
    #np.save('lemon_data/layer18_quan_conv.npy', quan_conv[0])

    n = np.size(gam)
    gam = gam.reshape(1, n, 1, 1)
    bias = bias.reshape(1, n, 1, 1)
    bn_out = (quan_conv + bias) * gam
    #np.save('lemon_data/layer18_quan_conv_add_mul.npy', 2 ** (gam_bit - 3) * bn_out[0])
    # bn_out=leaky1(bn_out)
    return bn_out


def conv_bias(a, activation_bit, quan_w, gam, bias):
    [batch_size, c_in, h, wi] = np.shape(a)
    aa1 = np.zeros([2, batch_size, c_in, h, wi])
    aa1[0] = a
    a = np.max(aa1, 0)
    aa2 = np.ones([2, batch_size, c_in, h, wi]) * (2 ** activation_bit - 1)
    aa2[0] = a
    a = np.min(aa2, 0)
    quan_a = np.rint(a)
    quan_a = np.float32(quan_a)
    #np.save('lemon_data/layer22_quan.npy', quan_a[0])
    # quan_w=np.float64(weight)
    [c_out, c_in, kh, kw] = np.shape(quan_w)
    # print(np.shape(quan_w))
    # quan_conv=lemon_batch(quan_a,quan_w,activation_bit)
    quan_conv = np.zeros([batch_size, c_out, h, wi])
    for i in range(c_out):
        quan_conv[:, i, :, :] = np.sum(quan_w[i:i + 1] * quan_a, 1)
    #np.save('lemon_data/layer22_quan_conv.npy', quan_conv[0])

    n = np.size(bias)
    bias = bias.reshape(1, n, 1, 1)
    layer_out = (quan_conv + bias) * gam
    #np.save('lemon_data/layer22_quan_conv_add_mul.npy', 2 ** (gam_bit - 3) * layer_out[0])
    return layer_out


# maxpool2d=torch.nn.MaxPool2d((2,2))
def maxpool2d_v2(X):
    [batch_size, c, h_in, w_in] = np.shape(X)
    h_out = int(h_in / 2)
    w_out = int(w_in / 2)
    Y = np.zeros([batch_size, c, h_out, w_out])
    for i in range(batch_size):
        for j in range(c):
            for k in range(h_out):
                for n in range(w_out):
                    Y[i, j, k, n] = np.max(X[i, j, 2 * k:2 * k + 2, 2 * n:2 * n + 2])
    return Y


def maxpool2d(X):
    [batch_size, c, h_in, w_in] = np.shape(X)
    # A = X[:,:,:,::2]
    # B = X[:,:,:,1::2]
    h_out = int(h_in / 2)
    w_out = int(w_in / 2)
    Y = np.zeros([4, batch_size, c, h_out, w_out])
    # Y[0] = A[:,:,::2,:]
    # Y[1] = A[:,:,1::2,:]
    # Y[2] = B[:,:,::2,:]
    # Y[3] = B[:,:,1::2,:]
    Y[0] = X[:, :, ::2, ::2]
    Y[1] = X[:, :, 1::2, ::2]
    Y[2] = X[:, :, ::2, 1::2]
    Y[3] = X[:, :, 1::2, 1::2]
    Z = np.max(Y, 0)
    return Z


def predict_transform(prediction, inp_dim_h, inp_dim_w, anchors, num_classes):
    [batch_size, _, grid_size_h, grid_size_w] = np.shape(prediction)
    stride = int(inp_dim_h / grid_size_h)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    grid_x = np.repeat(np.arange(grid_size_w).reshape(1, grid_size_w), grid_size_h, axis=0)
    grid_y = np.repeat(np.arange(grid_size_h).reshape(grid_size_h, 1), grid_size_w, axis=1)
    scaled_anchors = np.array([[a_w / stride, a_h / stride] for a_w, a_h in anchors])
    anchor_w = scaled_anchors[:, 0:1].reshape(3)
    anchor_h = scaled_anchors[:, 1:2].reshape(3)

    prediction = prediction.reshape(batch_size, num_anchors, bbox_attrs, grid_size_h, grid_size_w).transpose(0, 1, 3, 4,
                                                                                                             2)
    pre_t = np.zeros([batch_size, 5, bbox_attrs])
    for i_img in range(batch_size):
        i_pre = prediction[i_img]
        indexs = np.argwhere(i_pre[..., 4] >= 0)
        n_pre = np.shape(indexs)[0]
        n = np.min(np.array([n_pre, 5]))
        for i in range(n):
            index0 = indexs[i][0]
            index1 = indexs[i][1]
            index2 = indexs[i][2]
            pre_t[i_img, i, 0] = 1 / (1 + np.exp(-1 * i_pre[index0, index1, index2, 0])) + grid_x[index1, index2]
            pre_t[i_img, i, 1] = 1 / (1 + np.exp(-1 * i_pre[index0, index1, index2, 1])) + grid_y[index1, index2]
            pre_t[i_img, i, 2] = np.exp(i_pre[index0, index1, index2, 2]) * anchor_w[index0]
            pre_t[i_img, i, 3] = np.exp(i_pre[index0, index1, index2, 3]) * anchor_h[index0]
            pre_t[i_img, i, 4:] = 1 / (1 + np.exp(-1 * i_pre[index0, index1, index2, 4:]))

    pre_t[:, :, :4] *= stride
    return pre_t


def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x-
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y-
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x+
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y+
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    x1 = np.concatenate((np.repeat(b1_x1, np.size(b2_x1), axis=0).reshape(1, -1), b2_x1.reshape(1, -1)), axis=0)
    inter_rect_x1 = np.max(x1, 0)
    y1 = np.concatenate((np.repeat(b1_y1, np.size(b2_y1), axis=0).reshape(1, -1), b2_y1.reshape(1, -1)), axis=0)
    inter_rect_y1 = np.max(y1, 0)
    x2 = np.concatenate((np.repeat(b1_x2, np.size(b2_x2), axis=0).reshape(1, -1), b2_x2.reshape(1, -1)), axis=0)
    inter_rect_x2 = np.min(x2, 0)
    y2 = np.concatenate((np.repeat(b1_y2, np.size(b2_y2), axis=0).reshape(1, -1), b2_y2.reshape(1, -1)), axis=0)
    inter_rect_y2 = np.min(y2, 0)
    # Intersection area
    inter_w = np.zeros([2, np.size(inter_rect_x1)])
    inter_w[0] = inter_rect_x2 - inter_rect_x1 + 1
    inter_h = np.zeros([2, np.size(inter_rect_y1)])
    inter_h[0] = inter_rect_y2 - inter_rect_y1 + 1
    inter_area = np.max(inter_w, 0) * np.max(inter_h, 0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]  # Object confidence filtering

        # If none are remaining => process next image
        if not np.size(image_pred):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        n = np.shape(image_pred)[0]
        class_confs = np.max(image_pred[:, 5:], 1).reshape(n, 1)
        class_preds = np.zeros([n, 1])
        for i in range(n):
            class_preds[i] = np.where(image_pred[i, 5:] == np.max(image_pred[i, 5:]))[0]

        detections = np.concatenate((image_pred[:, :5], class_confs, class_preds), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        while np.shape(detections)[0]:
            large_overlap = bbox_iou(detections[0:1, :4], detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = np.sum(weights * detections[invalid, :4], 0) / np.sum(weights)
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            n_box = np.shape(keep_boxes)[0]
            output[image_i] = np.zeros([n_box, 7])
            for i in range(n_box):
                output[image_i][i] = keep_boxes[i]

    return output


# para=np.load('lemon_V4_para_v2.npy').item()
# img = np.load('data/img.npy')
#img = np.load('C:/HDocument/lemon/com_V2_1/img_torch.Size([100, 3, 288, 512])_.npy')
#print('Dell G3')
# print('Ultra96 v2')

img = cv2.imread('E:/python_project/prh_extract1/data_test/000002.jpg') #这是一张图片
print("完成图片读取")
print(type(img))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#display(Image.fromarray(img))
print('picture',5,'is being processed' )
img_resize = cv2.resize(img, (512,288), interpolation=cv2.INTER_NEAREST)
data = img_resize.transpose(2, 0, 1).reshape(1,3,288,512)



start = time.time()

#data = img[0:100]
#np.save('./knight_data/img.npy', data[0])
para = np.load('./knight_para_v0424_int.npy',allow_pickle=True).item()
para0 = para['0']
weight = para0['conv_w']
gam = para0['gam']
bias = para0['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
layer0_start = time.time()
layer0 = CM0(data, 8, weight, gam, bias) #quan_bit = 0
layer0_end = time.time()
print('Layer0: convolution and batch normalization. Execution time:', layer0_end - layer0_start)
print("layer0")
print(layer0)

"""
layer1_start = time.time()
layer1 = maxpool2d(layer0)
layer1_end = time.time()
print('Layer1: maxpooling. Execution time:',layer1_end-layer1_start)
"""

data = layer0
para2 = para['2']
weight = para2['conv_w']
gam = para2['gam']
bias = para2['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
layer2_start = time.time()
layer2 = CM2(data, 4, weight, gam, bias)
layer2_end = time.time()
print('Layer2: convolution and batch normalization. Execution time:', layer2_end - layer2_start)

"""
layer_start = time.time()
layer3 = maxpool2d(layer2)
layer_end = time.time()
print('Layer3: maxpooling. Execution time:',layer_end-layer_start)
"""

data = layer2
para2 = para['4']
weight = para2['conv_w']
gam = para2['gam']
bias = para2['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
layer_start = time.time()
layer4 = CM4(data, 4, weight, gam, bias)
layer_end = time.time()
print('Layer4: convolution and batch normalization. Execution time:', layer_end - layer_start)

"""
layer_start = time.time()
layer5 = maxpool2d(layer4)
layer_end = time.time()
print('Layer5: maxpooling. Execution time:',layer_end-layer_start)
"""

data = layer4
para2 = para['6']
weight = para2['conv_w']
gam = para2['gam']
bias = para2['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
layer_start = time.time()
layer6 = CM6(data, 4, weight, gam, bias)
layer_end = time.time()
print('Layer6: convolution and batch normalization. Execution time:', layer_end - layer_start)

"""
layer_start = time.time()
layer7 = maxpool2d(layer6)
layer_end = time.time()
print('Layer7: maxpooling. Execution time:',layer_end-layer_start)
"""

data = layer6
para2 = para['8']
weight = para2['conv_w']
gam = para2['gam']
bias = para2['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
layer_start = time.time()
layer9 = CM8(data, 4, weight, gam, bias)
layer_end = time.time()
print('Layer8: convolution and batch normalization. Execution time:', layer_end - layer_start)

'''
para19 = para['19']
gam = para19['gam']
gam = gam / 2 ** (gam_bit - 3)
layer9_19 = CM(data, 3, weight, gam, bias)
np.save('lemon_data/layer9_19_quan_conv_max_add_mul.npy', 2 ** (gam_bit - 3) * layer9_19[0])
'''
layer9_19 = layer9

data = layer9
para2 = para['10']
weight = para2['conv_w']
gam = para2['gam']
bias = para2['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
layer_start = time.time()
layer10 = CB10(data, 4, weight, gam, bias)
layer_end = time.time()
print('Layer10: convolution and batch normalization. Execution time:', layer_end - layer_start)

data = layer10
para2 = para['13']
weight = para2['conv_w']
gam = para2['gam']
bias = para2['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
layer_start = time.time()
layer13 = CB_v2_13(data, 4, weight, gam, bias)
layer_end = time.time()
print('Layer13: convolution and batch normalization. Execution time:', layer_end - layer_start)

"""
data = layer13
para2 = para['14']
weight = para2['conv_w'] 
a_scale = para2['a_scale']
w_scale = para2['w_scale']
bn_w = para2['bn_w']
bn_b = para2['bn_b']
bn_mean = para2['bn_mean']
bn_var = para2['bn_var']
layer_start = time.time()
layer14CB = CB(data, a_scale, 3, weight, w_scale, 3, bn_w, bn_b, bn_mean, bn_var)
layer14 = leaky1(layer14CB)
layer_end = time.time()
print('Layer14: convolution batch normalization and activation. Execution time:',layer_end-layer_start)


data = layer14
para2 = para['15']
weight = para2['conv_w']
a_scale = para2['a_scale']
w_scale = para2['w_scale']
conv_b = para2['conv_b']
layer_start = time.time()
layer15 = conv_bias(data, a_scale, 9, weight, w_scale, 3, conv_b)
layer_end = time.time()
print('Layer15: convolution. Execution time:',layer_end-layer_start)


mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319

mask = [int(x) for x in mask]

anchors = b [int(a) for a in anchors]
anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
anchors = [anchors[i] for i in mask]

inp_dim_h = 288
inp_dim_w = 512
num_classes = 12
layer16_start = time.time()
layer16 = predict_transform(layer15,inp_dim_h,inp_dim_w,anchors,num_classes)
layer16_end = time.time()
print('Layer16: yolo. Execution time:',layer16_end-layer16_start)
"""

data = layer13
para2 = para['18']
weight = para2['conv_w']
gam = para2['gam']
bias = para2['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
layer_start = time.time()
layer18 = CB_v2_18(data, 4, weight, gam, bias)
layer_end = time.time()
print('Layer18: convolution and batch normalization. Execution time:', layer_end - layer_start)

#para19 = para['19']
#bet = para19['bet']
#bet = bet / 2 ** (bet_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
#print('bet', bet)
layer19 = np.concatenate((layer18, layer9_19), axis=1)
#np.save('lemon_data/layer19.npy', 2 ** (gam_bit - 3) * layer19[0])

# upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
layer_start = time.time()
layer20 = np.repeat(np.repeat(layer19, 2, axis=2), 2, axis=3)
layer_end = time.time()
print('Layer20: upsampling. Execution time:', layer_end - layer_start)
#np.save('lemon_data/layer20.npy', 2 ** (gam_bit - 3) * layer20[0])

para21 = para['21']
weight = para21['conv_w']
gam = para21['gam']
bias = para21['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
data = layer20
layer_start = time.time()
layer21 = CB21(data, 4, weight, gam, bias)
layer_end = time.time()
print('Layer21: convolution and batch normalization. Execution time:', layer_end - layer_start)

data = layer21
para2 = para['22']
weight = para2['conv_w']
gam = para2['gam']
bias = para2['bias']
bias = bias / 2 ** (bias_bit - 13)
gam = gam / 2 ** (gam_bit - 3)
print('bias', np.max(bias), np.min(bias))
print('gam', np.max(gam), np.min(gam))
layer_start = time.time()
layer22 = conv_bias(data, 8, weight, gam, bias)
layer_end = time.time()
print(np.max(layer22), np.min(layer22))
print('Layer22: convolution. Execution time:', layer_end - layer_start)
layer22 = 2 ** layer22_dec_bit * layer22
layer22 = np.rint(layer22)
layer22 = layer22 / 2 ** layer22_dec_bit

mask = 0, 1, 2
anchors = 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319

mask = [int(x) for x in mask]

anchors = [int(a) for a in anchors]
anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
anchors = [anchors[i] for i in mask]

inp_dim_h = 288
inp_dim_w = 512
num_classes = 12
layer_start = time.time()
layer23 = predict_transform(layer22, inp_dim_h, inp_dim_w, anchors, num_classes)
layer_end = time.time()
print('Layer23: yolo. Execution time:', layer_end - layer_start)

# layer24 = np.concatenate((layer16,layer23),axis=1)
out = non_max_suppression(layer23)

# yolo16 = np.load('data/16_yolo.npy')
# yolo23 = np.load('data/23_yolo.npy')
# yolo = np.concatenate((yolo16,yolo23),axis=1)
# out = non_max_suppression(yolo)

batch_size = np.shape(out)[0]
pre_box = np.zeros([batch_size, 7])
for i in range(batch_size):
    if np.size(out[i]) > 2:
        pre_box[i] = out[i][0]

end = time.time()
print("Total Time: ", end - start, '\n')

# np.save('data/layer24.npy',layer24)
# yolo=np.load("C:/HDocument/lemon/com_V2_1/yolo_torch.Size([100, 2160, 17])_.npy")
# ground_truth=np.load("gt_torch.Size([8, 6])_.npy")

# ground_truth=np.load("image500/image_groundtruth/gt_torch.Size([500, 6])_.npy")
# ground_truth=ground_truth[0:1]
# print(ground_truth[0])





'''
#new ground_truth = np.load("C:/HDocument/lemon/com_V2_1/gt_torch.Size([100, 6])_.npy")
box_true = np.zeros([1, 4])
box_pre = np.zeros([1, 4])
cla_acc = np.zeros([batch_size])
iou = np.zeros([batch_size])

for i in range(batch_size):
    cla_true = ground_truth[i][1]
    box_true[0] = ground_truth[i][2:]
    pre = pre_box[i]
    cla_pre = pre[6]
    print('picture', i)
    print(' ground_truth', np.rint(ground_truth[i][2:]))
    if cla_pre == cla_true:
        cla_acc[i] = 1
        box_pre[0] = pre[:4]
        iou[i] = bbox_iou(np.rint(360 * box_true / 288), np.rint(360 * box_pre / 288))
        print(' prediction  ', np.rint(pre[:4]))
        print(' IoU:', iou[i])
    else:
        print(" Prediction is error!")
        print(' prediction  ', np.rint(pre[:4]))

print('')
acc = np.mean(cla_acc)
iou_mean = np.mean(iou)
print("class accuracy:", acc)
print("mean IoU:", iou_mean)


'''


layer23 = predict_transform(layer22,inp_dim_h,inp_dim_w,anchors,num_classes)
#new out = non_max_suppression(layer23)[0]
out = non_max_suppression(layer23)

print("----------layer22---------------------")
print(layer22)
print(np.shape(layer22))
print("----------layer23---------------------")
print(layer23)
print("-------------------------------")
print(np.size(out))
print(out)


'''
if np.size(out) >0.001:
    #ground_truth[i][2:] = 360*ground_truth[i][2:]/288
    result_rectangle = np.int16(np.rint(360*out[0][:4]/288))
    #box_true[0] = np.rint(ground_truth[i][2:])
    #box_pre[0] = result_rectangle
    #iou = bbox_iou(box_true,box_pre)
    #cv2.rectangle(img,(int(ground_truth[i][2]),int(ground_truth[i][3])),(int(ground_truth[i][4]),int(ground_truth[i][5])),(0,255,0),1)
    cv2.rectangle(img,(int(result_rectangle[0]),int(result_rectangle[1])),(int(result_rectangle[2]),int(result_rectangle[3])),(255,0,0),1)
    img = cv2.putText(img,class_num2name(out[0][6]),(int(result_rectangle[0]),int(result_rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    #display(Image.fromarray(img))
    #print('IoU:',float('%.2f'%(100*iou)),'%')
    cv2.imshow("image", img)  # 显示图片，后面会讲解
    cv2.waitKey(0)  # 等待按键
else:
    print('No targets have been detected.')

'''


