# 箱体检测与尺寸估计（传统视觉 + 点云）

本项目基于 `.mat` 格式的深度相机数据（幅值图、距离图、点云）实现：

- 地面平面检测（RANSAC）
- 箱顶平面检测（RANSAC）
- 箱体高/长/宽估计
- 2D/3D 结果可视化
- Streamlit 参数调节页面（浏览器交互调参）

---

## 1. 项目结构

- `main.py`：离线主流程脚本
- `streamlit_app.py`：浏览器 UI（可选文件、可调参数、可视化结果）
- `data_utils.py`：数据读取与有效点提取
- `ransac.py`：平面拟合与 RANSAC
- `mask_utils.py`：掩码构建与清理
- `geometry_utils.py`：尺寸估计与角点估计
- `visualization.py`：2D/3D 可视化函数
- `DataRead.py`：批量检查 `.mat` 文件内容
- `data/`：数据目录（放置 `.mat` 文件）

---

## 2. 数据格式要求

`.mat` 文件中需要包含以下字段（前缀匹配）：

- `amplitudes*`：幅值图（2D）
- `distances*`：距离图（2D）
- `cloud*`：点云（`H x W x 3`）

程序会自动提取这三类字段并校验形状一致性。

---

## 3. 环境安装

建议 Python 3.10+。

```bash
pip install -r requirements.txt
```

---

## 4. 运行方式

### 4.1 运行离线脚本

```bash
python3 main.py
```

该脚本默认读取 `data/example1kinect.mat`。

### 4.2 运行 Streamlit 调参界面

```bash
streamlit run streamlit_app.py
```

在页面中可以：

- 选择本地 `data/` 下的 `.mat` 文件，或上传 `.mat` 文件
- 调整 RANSAC 和掩码清理参数
- 查看尺寸估计和可视化结果

---

## 5. 可调关键参数说明

在 `streamlit_app.py` 页面中可调：

- `Floor threshold`：地面平面内点距离阈值
- `Box top threshold`：箱顶平面内点距离阈值
- `Max iterations`：RANSAC 最大迭代次数
- `Min floor component size`：地面最小连通域面积
- `Max internal hole size`：地面内部可填充空洞上限
- `Length/width estimation method`：长宽估计方法（`pca` / `simple`）

---

## 6. 结果说明

最终可得到：

- 箱体高度（由地面平面和箱顶平面距离估计）
- 箱体长宽（基于箱顶点云，默认 PCA）
- 箱顶四角点（近似）
- 2D 分割叠加图与方向标签（`top/bottom/left/right`）

---

## 7. 常见问题

- **报错 “No valid points”**：点云中 `z != 0` 的有效点不足。
- **箱顶检测不稳定**：尝试调大 `Max iterations`，并微调两个阈值。
- **地面掩码过碎**：适当增大 `Min floor component size`。
- **箱体孔洞被误填**：适当减小 `Max internal hole size`。
