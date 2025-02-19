import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 读取 CSV 文件
file_path = 'tmp/preds_surrogate_hidden.pkl'  # 请替换为你的实际文件路径
df = pd.read_pickle(file_path)

# 删除 rad_atom_hiddens 仅包含 [0.] 的行
df = df[df["rad_atom_hiddens"].apply(lambda x: not (len(x) == 1 and x[0] == 0.))]

# 获取 mol_hiddens 和 rad_atom_hiddens 的特征数据
mol_features = np.stack(df["mol_hiddens"].values)
rad_features = np.stack(df["rad_atom_hiddens"].values)

# 进行 t-SNE 降维
tsne_mol = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_rad = TSNE(n_components=2, perplexity=30, random_state=42)

mol_2d = tsne_mol.fit_transform(mol_features)
rad_2d = tsne_rad.fit_transform(rad_features)

# 获取特定的 smiles 索引
highlight_smiles = ["[O]CC(Cl)(Cl)Cl", "[O]CC(F)(F)F"]
smiles_list = df["smiles"].tolist()
highlight_indices = [i for i, smi in enumerate(smiles_list) if smi in highlight_smiles]

# 画图函数
def plot_tsne(tsne_data, title, highlight_indices):
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c='gray', alpha=0.5, label="Other Data")
    
    # 标注特定 smiles
    for idx in highlight_indices:
        plt.scatter(tsne_data[idx, 0], tsne_data[idx, 1], c='red', edgecolors='black', s=100, label=smiles_list[idx])
        plt.text(tsne_data[idx, 0], tsne_data[idx, 1], smiles_list[idx], fontsize=10, color='black')
    
    plt.title(title)
    plt.legend()
    plt.savefig(title+'.jpg')

# 绘制 t-SNE 结果
plot_tsne(mol_2d, "t-SNE Visualization of mol_hiddens", highlight_indices)
plot_tsne(rad_2d, "t-SNE Visualization of rad_atom_hiddens", highlight_indices)

