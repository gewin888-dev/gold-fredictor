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

## 复制到其他 Mac

当前版本支持复制到其他 Mac 后以便携模式首次启动：

- App 包内包含 Python 源码和前端页面。
- 首次启动会在 `~/Library/Application Support/gold-fredictor-app/runtime/` 创建 `.venv`、`.env`、数据库和日志目录。
- 首次启动需要目标电脑已安装 `python3`，并允许联网安装 `requirements.txt` 里的依赖。
- 个人 API Key 不会打进 App；需要在目标电脑的配置页面或 `.env` 中填写。
- 未做 Apple Developer ID 签名/公证时，其他电脑首次打开可能需要右键选择“打开”。

端口可通过环境变量覆盖：

```bash
GOLD_FREDICTOR_API_PORT=8010 GOLD_FREDICTOR_DASHBOARD_PORT=8510 npm run app
```

App 可以移动到桌面运行。启动时会按以下顺序查找项目根目录：

1. `GOLD_FREDICTOR_PROJECT_ROOT`
2. App 原始打包目录外层的项目目录
3. 当前用户目录下的 `gold fredictor`

如果项目目录也被移动，需要指定项目根目录：

```bash
GOLD_FREDICTOR_PROJECT_ROOT="/Users/gewin/gold fredictor" \
  "dist-app/mac-arm64/Gold Fredictor.app/Contents/MacOS/Gold Fredictor"
```

模拟纯便携模式：

```bash
GOLD_FREDICTOR_PORTABLE_MODE=1 \
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
