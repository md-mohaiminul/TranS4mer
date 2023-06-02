import torch

# from TimeSformer.timesformer.models.vit import TimeSformer
#
# model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='joint_space_time')
#                     #pretrained_model='/playpen-storage/mmiemon/ego4d/NLQ/TimeSformer/TimeSformer_divST_8x32_224_K400.pyth')
#
# dummy_video = torch.randn(2, 3, 8, 224, 224) # (batch x channels x frames x height x width)
#
# pred = model(dummy_video,) # (2, 400)
#
# print(pred.shape)

# from timm.models import create_model
#
# m = create_model('vit_small_patch32_224', pretrained=True, num_classes=0, attention_type = 'cls').cuda()

# x = torch.randn(190, 3, 224, 224).cuda()
# o1 = m(x, s=19)
# print(o1.shape)
#
# m = create_model('vit_small_patch32_224', pretrained=True, num_classes=0, attention_type = 'spatial').cuda()
# o2 = m(x, s=19)
# print(o2.shape)
#
# m = create_model('vit_small_patch32_224', pretrained=True, num_classes=0, attention_type = 'joint').cuda()
# o3 = m(x, s=19)
# print(o3.shape)
#
# m = create_model('vit_small_patch32_224', pretrained=True, num_classes=0, attention_type = 's4').cuda()
# o4 = m(x, s=19)
# print(o4.shape)
#
# print(torch.sum(o1-o2))
# print(torch.sum(o2-o3))
# print(torch.sum(o2-o4))

from s4 import S4

model = S4(d_model=1024, l_max=1024, bidirectional=True, postact='glu', dropout=0.1, transposed=True)
model.cuda()
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)
        # D torch.Size([1, 1024])
        # kernel.kernel.C torch.Size([2, 1024, 32, 2])
        # output_linear.0.weight torch.Size([2048, 1024, 1])
        # output_linear.0.bias torch.Size([2048])

x = torch.randn(16, 1024, 512).cuda()
o, _ = model(x)
print(o.shape)


# import torch
# from gated_state_spaces_pytorch import GSS
#
# gss = GSS(
#     dim = 512,                  # dimension
#     dim_expansion_factor = 4,   # hidden dimension (expansion factor x dim) = 2048
#     dss_kernel_N = 512,
#     dss_kernel_H = 256
# )
#
# for name, param in gss.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape)
#
# x = torch.randn(1, 1024, 512)
#
# out = gss(x) # (1, 65536, 512)
# print(out.shape)