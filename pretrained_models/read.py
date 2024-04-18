import torch
pthfile = r'./DiT-XL-2-512x512.pt'  # .pth文件的路径
model = torch.load(pthfile, torch.device('cpu'))  # 设置在cpu环境下查询
print('type:')
print(type(model))  # 查看模型字典长度
print('length:')
print(len(model))
print('key:')
for k in model.keys():  # 查看模型字典里面的key
    print(k)
print('value:')
for k in model:  # 查看模型字典里面的value
    print(k, model[k])

# import torch
#
# model_dict = torch.load('DiT-XL-2-512x512.pt', map_location='cpu')
# for name, para in model_dict.items():
#     print(name, ':', para.size())