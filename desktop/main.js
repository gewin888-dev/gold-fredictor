const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const DASHBOARD_URL = "http://127.0.0.1:8501";
const API_HEALTH_URL = "http://127.0.0.1:8000/health";
let mainWindow = null;
let serviceProcess = null;

function isRunnableProjectRoot(root) {
  return (
    root &&
    fs.existsSync(path.join(root, "scripts", "start_local_app.sh")) &&
    fs.existsSync(path.join(root, ".venv", "bin", "python"))
  );
}

function projectRoot() {
  if (app.isPackaged) {
    const candidates = [
      process.env.GOLD_FREDICTOR_PROJECT_ROOT,
      path.resolve(process.resourcesPath, "../../../../.."),
      path.join(process.resourcesPath, "app")
    ];
    const root = candidates.find(isRunnableProjectRoot);
    if (root) {
      return root;
    }
    throw new Error(
      "找不到可运行的项目目录。请把 App 放在本项目 dist-app 目录下运行，或设置 GOLD_FREDICTOR_PROJECT_ROOT。"
    );
  }
  return path.resolve(__dirname, "..");
}

function waitFor(url, timeoutMs = 60000) {
  const startedAt = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(url, (res) => {
        res.resume();
        if (res.statusCode >= 200 && res.statusCode < 500) {
          resolve();
          return;
        }
        retry();
      });
      req.on("error", retry);
      req.setTimeout(3000, () => {
        req.destroy();
        retry();
      });
    };
    const retry = () => {
      if (Date.now() - startedAt > timeoutMs) {
        reject(new Error(`Timed out waiting for ${url}`));
        return;
      }
      setTimeout(tick, 1000);
    };
    tick();
  });
}

function startServices() {
  const root = projectRoot();
  const script = path.join(root, "scripts", "start_local_app.sh");
  serviceProcess = spawn("/bin/bash", [script, "--no-open"], {
    cwd: root,
    env: {
      ...process.env,
      GOLD_FREDICTOR_NO_OPEN: "1",
      NO_PROXY: "127.0.0.1,localhost",
      no_proxy: "127.0.0.1,localhost"
    },
    stdio: ["ignore", "pipe", "pipe"]
  });

  serviceProcess.stdout.on("data", (chunk) => {
    console.log(`[service] ${chunk.toString().trim()}`);
  });
  serviceProcess.stderr.on("data", (chunk) => {
    console.error(`[service] ${chunk.toString().trim()}`);
  });
  serviceProcess.on("exit", (code) => {
    if (code && code !== 0) {
      console.error(`Service launcher exited with ${code}`);
    }
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    title: "Gold Fredictor",
    backgroundColor: "#f6f8fb",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  mainWindow.loadURL(DASHBOARD_URL);
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

async function boot() {
  try {
    startServices();
    await waitFor(API_HEALTH_URL, 60000);
    await waitFor(DASHBOARD_URL, 60000);
    createWindow();
  } catch (error) {
    dialog.showErrorBox(
      "Gold Fredictor 启动失败",
      `${error.message}\n\n请确认 .venv 已安装依赖，并可运行 ./scripts/start_local_app.sh。`
    );
    app.quit();
  }
}

app.whenReady().then(boot);

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0 && mainWindow === null) {
    createWindow();
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
