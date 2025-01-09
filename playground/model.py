import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

# 自定义的自注意力层
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        # 每个头的维度
        self.head_dim = embed_dim // num_heads
        
        # 查询、键、值的线性变换
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # 输出的线性变换
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]  # 批量大小
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # 计算查询、键、值的线性变换
        queries = self.query(query)
        keys = self.key(keys)
        values = self.value(values)
        
        # 将查询、键、值分成多头
        queries = queries.view(N, query_len, self.num_heads, self.head_dim)
        keys = keys.view(N, key_len, self.num_heads, self.head_dim)
        values = values.view(N, value_len, self.num_heads, self.head_dim)
        
        # 交换维度，得到形状为 (N, num_heads, seq_len, head_dim)
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)
        
        # 计算缩放点积注意力分数
        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))  # (N, num_heads, query_len, key_len)
        
        # 如果有 mask，应用它
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        # 计算注意力权重
        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=-1)  # (N, num_heads, query_len, key_len)
        
        # 应用注意力权重到值上
        out = torch.matmul(attention, values)  # (N, num_heads, query_len, head_dim)
        
        # 恢复形状
        out = out.permute(0, 2, 1, 3).contiguous()  # (N, query_len, num_heads, head_dim)
        out = out.view(N, query_len, self.num_heads * self.head_dim)  # (N, query_len, embed_dim)
        
        # 通过线性层输出
        out = self.fc_out(out)  # (N, query_len, embed_dim)
        
        return out

# 示例模型：将自注意力层应用到输入
# class SimpleTransformer(nn.Module):
#     def __init__(self, embed_dim=512, num_heads=8, num_classes=10):
#         super(SimpleTransformer, self).__init__()
        
#         # 线性嵌入层，将输入的每个图像展平后映射到 embed_dim 维空间
#         self.embedding = nn.Linear(28*28, embed_dim)
        
#         # 定义自注意力层
#         self.attention = SelfAttention(embed_dim, num_heads)
        
#         # 分类层
#         self.fc = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         # 展平图像并映射到 embed_dim
#         x = self.embedding(x.view(x.shape[0], -1))  # (batch_size, embed_dim)
        
#         # 添加额外的维度以适应注意力机制的输入
#         x = x.unsqueeze(1)  # (batch_size, 1, embed_dim) 这里将图像的每个像素作为一个时间步输入
        
#         # 应用自注意力层
#         x = self.attention(x, x, x)  # 传入相同的值、键和查询
        
#         # 只取最后的输出，进行分类
#         x = x.mean(dim=1)  # 聚合序列输出，(batch_size, embed_dim)
#         x = self.fc(x)  # (batch_size, num_classes)
        
#         return x

class SimpleTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_classes=10, patch_size=2):
        super(SimpleTransformer, self).__init__()
        self.patch_size = patch_size
        # 线性嵌入层，将输入的每个patch映射到embed_dim维空间
        self.embedding = nn.Linear(patch_size*patch_size, embed_dim)
        
        # 定义自注意力层
        self.attention = SelfAttention(embed_dim, num_heads)
        
        # 分类层
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 将图像切分成patch
        # x: (batch_size, channels, height, width)
        # 确保图像尺寸是patch_size的整数倍
        height, width = x.shape[2], x.shape[3]
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Height and width must be divisible by patch_size"
        
        # 将图像切分成patch并重塑
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(x.shape[0], -1, self.patch_size*self.patch_size)  # (batch_size, num_patches, self.patch_size*self.patch_size)
        
        # 映射到embed_dim
        x = self.embedding(x)  # (batch_size, num_patches, embed_dim)
        
        # 添加额外的维度以适应注意力机制的输入
        x = x.transpose(0, 1)  # (num_patches, batch_size, embed_dim)
        
        # 应用自注意力层
        x = self.attention(x, x, x)  # 传入相同的值、键和查询
        
        # 只取最后的输出，进行分类
        x = x.transpose(0, 1).mean(dim=1)  # (batch_size, embed_dim)
        x = self.fc(x)  # (batch_size, num_classes)
        
        return x