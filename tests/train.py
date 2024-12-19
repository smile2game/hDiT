import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from model import SimpleTransformer,SelfAttention

# 加载数据
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
model = SimpleTransformer(embed_dim=512, num_heads=8, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练
for epoch in range(10):
    epoch_loss = 0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
        for data, target in pbar:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # 累加损失并更新进度条
            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss/len(pbar))  # 显示每个 epoch 的平均损失
    
    print(f"Epoch {epoch+1}: Average Loss {epoch_loss/len(train_loader)}")

# 保存模型权重
model_weights_path = 'simple_transformer_weights.pth'
torch.save(model.state_dict(), model_weights_path)
print(f"Model weights saved to {model_weights_path}")



