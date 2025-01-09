import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from model import SimpleTransformer, SelfAttention


import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

# 确保模型架构与保存权重时的架构相同
model = SimpleTransformer(embed_dim=512, num_heads=8, num_classes=10)

# 加载模型权重
model_weights_path = 'simple_transformer_weights.pth'
model.load_state_dict(torch.load(model_weights_path))
print(f"Model weights loaded from {model_weights_path}")

# 将模型设置为评估模式
model.eval()

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

# 从数据集中读取一个批次的图像数据
data_iter = iter(train_loader)
images, labels = next(data_iter)  # 获取一个批次的图像和标签, batch_size = 64

# 确保模型处于评估模式
model.eval()

# 将模型设置为不计算梯度，以减少内存消耗和计算时间
with torch.no_grad():
    # 遍历批次中的每张图片
    correct_count = 0  # 正确预测的数量
    total_count = images.size(0)  # 批次中的总图片数量
    for i in range(images.size(0)):  # images.size(0) 是批次大小
        input_data = images[i].unsqueeze(0)  # 将图像维度从 (28, 28) 扩展为 (1, 28, 28)
        
        # 进行推理
        output = model(input_data)
        predicted_class = output.argmax(dim=1)  # 获取预测类别
        print(f"Image {i+1}, Predicted class: {predicted_class.item()}")  # 输出预测结果
        
        # 打印真实标签
        true_class = labels[i].item()
        print(f"Image {i+1}, True class: {true_class}")
        print("-" * 40)  # 打印分隔线以区分不同的图片输出

        if predicted_class == true_class:
            correct_count += 1
    accuracy = correct_count / total_count
    print(f"Accuracy for this batch: {accuracy:.2f}")