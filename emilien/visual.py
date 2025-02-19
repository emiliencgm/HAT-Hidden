import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

file = "tmp/input_ffnn.pkl"
# 读取pkl文件
df = pd.read_pickle(file)
# remove = [1,2,3,4,5,6]
remove = []
# df = df.drop(df.columns[remove], axis=1)

# 假设前两列是索引和rxn_id，最后一列是标签，其他列为特征
features = df.iloc[:, 1:-1]  # 选择特征列
labels = df.iloc[:, -1]  # 获取标签列

# 处理特征列：如果某个特征是向量（列数组），直接提取为numpy数组
def extract_feature_vector(value):
    if isinstance(value, np.ndarray):  # 如果是numpy数组，直接返回
        return value
    elif isinstance(value, list):  # 如果是列表，也可以转换为numpy数组
        return np.array(value)
    return np.array([value])  # 其他情况返回一个数值数组

# 将所有特征值转换为数值向量
features = features.applymap(extract_feature_vector)

# 将每一行的特征列拼接成一个大的向量
def flatten_features(row):
    return np.concatenate([x for x in row])  # 拼接所有特征

# 拼接每行的特征并转为numpy数组
features_array = np.array(features.apply(flatten_features, axis=1).tolist())
print("@@@", features_array.shape)

# 进行 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(features_array)

# 绘制散点图
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="Label Value")
plt.title("t-SNE Visualization of Feature Distribution")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.savefig(file+str(remove)+'In-House'+'.jpg')
