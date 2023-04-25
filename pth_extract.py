import torch
import numpy as np
import math

pthfile = r'E:/课题组工作/量化网络/量化任务/new version/Tiny-YOLO-LSQ_modified/exps/lsq_ckpt/lsq_ckpt_94.pth'

model = torch.load(pthfile, torch.device('cpu'))


file = "./out_file.txt"#输出文件路径

with open(file,'w') as outfile:
    for k in model:
        outfile.write(k)
        outfile.write('\n')
        outfile.write(str(model[k]))
        outfile.write('\n')

outfile.close()


eps = 1e-5
M = 13


def count_bias(bn_m, bn_b, bn_var, bn_w, sa, sw):
    # all is np array
    bn_m_np = bn_m.numpy()
    bn_b_np = bn_b.numpy()
    bn_var_np = bn_var.numpy()
    bn_w_np = bn_w.numpy()
    sa_np = sa.numpy()
    sw_np = sw.numpy()  # float

    bias_np = np.rint((- bn_m_np + bn_b_np * np.sqrt(bn_var_np + eps) / bn_w_np) / (sa_np * sw_np))
    return bias_np.astype(np.int16)

def count_bias1(conv_bias, sa, sw):
    # all is np array
    conv_bias_np = conv_bias.numpy()
    sa_np = sa.numpy()
    sw_np = sw.numpy()  # float

    bias_np = np.rint((conv_bias_np) / (sa_np * sw_np))
    return bias_np.astype(np.int16)


def count_gam(sa, sw, bn_w, bn_var, sa1, M):
    bn_var_np = bn_var.numpy()
    bn_w_np = bn_w.numpy()
    sa_np = sa.numpy()
    sw_np = sw.numpy()  # float
    sa1_np = sa1.numpy()

    gam_np = np.rint((sa_np * sw_np * bn_w_np / (np.sqrt(bn_var_np + eps) * sa1_np)) * pow(2, M))
    return gam_np.astype(np.int16)

def count_gam1(sa, sw, M):
    sa_np = sa.numpy()
    sw_np = sw.numpy()  # float

    gam_np = np.rint(sa_np * sw_np  * pow(2, M))
    return gam_np.astype(np.int16)

def count_weight(sw, weight):
    sw_np = sw.numpy()  # float
    weight_np = weight.numpy()  # output_channel -- input_channel -- height -- width
    weight_quan_np = np.clip(np.rint(weight_np / sw_np), -3, 3)
    return weight_quan_np.astype(np.int16)  # but need int 3bits

#浮点数转化为整数


def get_param(model, bias_file, gam_file, weight_file):
    with open(bias_file, "w") as outfile1, open(gam_file, "w") as outfile2, open(weight_file, "w") as outfile3:

        weight_key = []
        sw_key = []
        sa_key = []
        bn_w_key = []
        bn_b_key = []
        bn_m_key = []
        bn_var_key = []
        for i in key_num:
            weight_key.append("module_list." + i + ".conv_" + i + ".weight")
            sw_key.append("module_list." + i + ".conv_" + i + ".quan_w_fn.s")
            sa_key.append("module_list." + i + ".conv_" + i + ".quan_a_fn.s")
            bn_w_key.append("module_list." + i + ".batch_norm_" + i + ".weight")
            bn_b_key.append("module_list." + i + ".batch_norm_" + i + ".bias")
            bn_m_key.append("module_list." + i + ".batch_norm_" + i + ".running_mean")
            bn_var_key.append("module_list." + i + ".batch_norm_" + i + ".running_var")

        for num in range(len(key_num)):
            outfile1.write("conv" + key_num[num] + ":\n")
            if num < len(key_num) - 1:
                outfile1.write(str(
                    count_bias(model[bn_m_key[num]], model[bn_w_key[num]], model[bn_var_key[num]], model[bn_w_key[num]],
                               model[sa_key[num]], model[sw_key[num]])))
                bias_tmp = count_bias(model[bn_m_key[num]], model[bn_w_key[num]], model[bn_var_key[num]], model[bn_w_key[num]],
                           model[sa_key[num]], model[sw_key[num]])
                outfile1.write("\n")
            else:
                outfile1.write(str(
                    count_bias1(model["module_list.22.conv_22.bias"],
                               model[sa_key[num]], model[sw_key[num]])))
                bias_tmp = count_bias1(model["module_list.22.conv_22.bias"],
                               model[sa_key[num]], model[sw_key[num]])
                outfile1.write("\n")




            outfile2.write("conv" + key_num[num] + ":\n")
            if num < len(key_num) - 1:
                outfile2.write(str(
                    count_gam(model[sa_key[num]], model[sw_key[num]], model[bn_w_key[num]], model[bn_var_key[num]],
                              model[sa_key[num + 1]], M)))
                gam_tmp = count_gam(model[sa_key[num]], model[sw_key[num]], model[bn_w_key[num]], model[bn_var_key[num]],
                              model[sa_key[num + 1]], M)
                outfile2.write("\n")
            else:
                outfile2.write(str(
                    count_gam1(model[sa_key[num]], model[sw_key[num]], M)))
                gam_tmp = count_gam1(model[sa_key[num]], model[sw_key[num]],  M)
                outfile2.write("\n")


            outfile3.write("conv" + key_num[num] + ":\n")

            outfile3.write(str(count_weight(model[sw_key[num]], model[weight_key[num]])))
            outfile3.write("\n")

            quan_w = count_weight(model[sw_key[num]], model[weight_key[num]])

            globals()["para"+key_num[num]] = {'conv_w':quan_w, 'bias':bias_tmp, 'gam':gam_tmp}
            para[key_num[num]]= globals()["para"+key_num[num]]




        outfile1.close()
        outfile2.close()
        outfile3.close()





bias_file = "./bias.txt"
gam_file = "./gam.txt"
weight_file = "./weight.txt"


key_num = ["0", "2", "4", "6", "8", "10", "13", "18", "21", "22"]  # 10 conv

for num in range(len(key_num)):
    globals()["para"+key_num[num]] = {}


para={'0':para0,'2':para2,'4':para4,'6':para6,'8':para8,'10':para10,'13':para13,'18':para18,'21':para21,'22':para22}


get_param(model, bias_file, gam_file, weight_file)
print(para)
np.save('knight_para_v0424_int.npy',para)



