# 📸 Lanmei AI Tools Station (蓝梅 AI 工具站)

这是一个功能强大的 AI 图像处理工具站，主要包含两个核心模块：
1.  **ExifTool (元数据清理器)**: 清理、修改图片 EXIF 信息，抹除 AIGC 痕迹，增加胶片颗粒感。
2.  **FusionTool (智能叠图/人像融合)**: 采用 SIFT 特征对齐、BiSeNet 面部分割、泊松融合与色彩协调技术，实现高质量的人像与场景融合。

---

## 🚀 快速开始

### 1. 环境准备
*   **Python**: 3.10+ (推荐 3.14)
*   **Node.js**: 18.0+
*   **ONNX 模型**: 确保 `models/bisenet/resnet18.onnx` 文件存在。

### 2. 本地运行 (开发模式)

#### 后端 (FastAPI)
```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动后端
python3 -m backend.main
```
后端服务将运行在 `http://localhost:8000`。

#### 前端 (Vite + React)
```bash
cd web
npm install
npm run dev
```
前端服务将运行在 `http://localhost:5173`。

---

### 3. 服务器部署 (Ubuntu 22.04 / 云服务器)

我们提供了一键启动脚本 `start_all.sh`，它会自动处理依赖安装、环境验证及服务监控（守护进程模式）。

#### 步骤：
1.  将项目代码上传到服务器。
2.  赋予脚本执行权限：
    ```bash
    chmod +x start_all.sh
    ```
3.  运行一键启动脚本：
    ```bash
    ./start_all.sh
    ```
#### 部署特性：
*   **全自动**: 自动创建 `.venv` 并安装前后端依赖。
*   **公网访问**: 自动绑定到 `0.0.0.0`，外网可直接访问。
*   **高可用**: 内置守护进程监控，服务意外崩溃会自动重启。
*   **代理转发**: 前端 Vite 自动配置了 `/api` 代理，无需额外配置 Nginx 即可通过前端端口访问 API。

---

## 🛠 技术栈
*   **Backend**: FastAPI, OpenCV, ONNX Runtime (CPU/CUDA/CoreML)
*   **Frontend**: React, Arco Design, Vite, Axios
*   **Logic**: SIFT 对齐, BiSeNet 分割, Poisson Blending, LAB 色彩转换

## 📝 注意事项
*   本项目目前已移除 `torch` 依赖，采用**纯 ONNX 推理**以减小部署体积并提升启动速度。
*   默认数据存储在项目根目录的 `data/` 文件夹下。

---
© 2026 Lanmei AI Studio. All rights reserved.
