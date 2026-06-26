# 桌面 App

桌面 App 使用 Electron 包装现有 FastAPI + Streamlit 系统。业务逻辑仍在 Python 服务中，App 负责一键启动本地服务并打开独立窗口。

## 本地启动

```bash
npm install
npm run app
```

如果 Electron 二进制下载长时间无响应，可以临时使用镜像源：

```bash
ELECTRON_MIRROR=https://npmmirror.com/mirrors/electron/ npm install
```

App 启动后会执行：

```bash
./scripts/start_local_app.sh --no-open
```

然后打开 `http://127.0.0.1:8501`。

## 打包

```bash
npm run app:dist
```

产物输出到 `dist-app/`。当前打包结果是本机项目启动器：默认从项目根目录读取 `.venv`、`.env` 和本地数据库，不会把个人密钥、虚拟环境或数据库打进 App 包。

App 可以移动到桌面运行。启动时会按以下顺序查找项目根目录：

1. `GOLD_FREDICTOR_PROJECT_ROOT`
2. App 原始打包目录外层的项目目录
3. 当前用户目录下的 `gold fredictor`

如果项目目录也被移动，需要指定项目根目录：

```bash
GOLD_FREDICTOR_PROJECT_ROOT="/Users/gewin/gold fredictor" \
  "dist-app/mac-arm64/Gold Fredictor.app/Contents/MacOS/Gold Fredictor"
```

打包前建议确认：

```bash
.venv/bin/python scripts/manage.py health
npm run app
```

## 故障排查

```bash
.venv/bin/python scripts/manage.py health
.venv/bin/python scripts/manage.py config --format table
./scripts/start_local_app.sh
```
