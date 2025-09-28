import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 加载数据
file_path = r"C:\Users\李旭\Downloads\train.csv"
data = pd.read_csv(file_path)

# 假设最后一列是目标变量，其余是特征
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(X).to(device)
y_tensor = torch.FloatTensor(y).to(device)

print(f"数据形状: X{X.shape}, y{y.shape}")


# 2. 定义线性模型
class LinearModel(nn.Module):
    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x).squeeze()


# 3. 初始化模型和优化器
model = LinearModel(X.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 训练并记录变化
epochs = 1000
losses = []
weights_history = []

for epoch in range(epochs):
    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失和权重
    losses.append(loss.item())

    # 记录当前权重
    with torch.no_grad():
        current_weights = model.linear.weight.detach().cpu().numpy().flatten()
        weights_history.append(current_weights)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

# 5. 可视化
plt.figure(figsize=(12, 4))

# 损失变化
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Loss during Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# 权重变化
plt.subplot(1, 2, 2)
weights_array = np.array(weights_history)
for i in range(weights_array.shape[1]):
    plt.plot(weights_array[:, i], label=f'w{i + 1}')
plt.title('Weights during Training')
plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 打印最终参数
print("\n最终权重:", model.linear.weight.detach().cpu().numpy())
print("最终偏置:", model.linear.bias.detach().cpu().item())