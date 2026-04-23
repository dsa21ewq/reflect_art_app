# Reflect Art App

一个可本地运行的交互式 Streamlit 原型，用于模拟：

- 平面纸张 `z = 0`
- 有限高竖直圆柱镜侧面
- 固定观察点
- 正向模拟 / 逆向设计 / 双向验证

## 项目结构

```text
reflect_art_app/
  app.py
  geometry.py
  imaging.py
  metrics.py
  utils.py
  requirements.txt
  README.md
```

## 几何模型

纸面平面：

```text
z = 0
```

圆柱面点：

```text
M(theta, h) = (xc + R cos(theta), yc + R sin(theta), h)
```

圆柱外法向：

```text
n(theta) = (cos(theta), sin(theta), 0)
```

观察点：

```text
E = (xe, ye, ze)
```

在镜面点处，用逆向追迹方式：

1. 从观察点指向镜面点，得到方向 `r = (M - E) / ||M - E||`
2. 由反射定律求逆向反射方向

```text
q = r - 2 (r · n) n
```

3. 与纸面 `z = 0` 求交，得到纸面点 `P`

```text
P = M + lambda q,  lambda = -h / qz
```

若 `qz >= 0`，说明无法打到纸面；若 `P` 超出 A4，则标记越界。

## 本地运行

### 1. 安装依赖

```bash
cd /Users/yourname/././.
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 启动 app

```bash
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 将本目录推送到 GitHub 仓库
2. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)
3. 选择 `New app`
4. 选择仓库、分支和主文件：

```text
app.py
```

5. 点击 `Deploy`

本项目不依赖外部私密数据，也不需要额外 secrets，直接部署即可。

## 功能说明

- `正向模拟`
  - 输入纸面图案
  - 计算观察者看到的镜面图案

- `逆向设计`
  - 输入目标镜面图案
  - 反求纸面应打印的图案
  - 并再正向模拟验证效果

- `双向验证`
  - 同时展示原纸面、目标镜面、原纸面正向镜面、反求纸面、验证镜面

## 默认参数

- A4: `210 mm × 297 mm`
- 圆柱半径: `20 mm`
- 圆柱高度: `120 mm`
- 圆柱中心: `(105, 150) mm`
- 观察点: `(105, -180, 320) mm`
- 采样数: `300 × 300`


