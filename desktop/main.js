const { app, BrowserWindow, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const API_PORT = process.env.GOLD_FREDICTOR_API_PORT || "8000";
const DASHBOARD_PORT = process.env.GOLD_FREDICTOR_DASHBOARD_PORT || "8501";
const API_ORIGIN = `http://127.0.0.1:${API_PORT}`;
const DASHBOARD_URL = `http://127.0.0.1:${DASHBOARD_PORT}`;
const API_HEALTH_URL = `${API_ORIGIN}/health`;
const AI_CHAT_URL = `${API_ORIGIN}/ai/ui`;
let mainWindow = null;
const childWindows = new Set();
let serviceProcess = null;
let serviceRuntimeDir = null;

function isRunnableProjectRoot(root) {
  return (
    root &&
    fs.existsSync(path.join(root, "scripts", "start_local_app.sh")) &&
    fs.existsSync(path.join(root, ".venv", "bin", "python"))
  );
}

function hasPackagedSource(root) {
  return (
    root &&
    fs.existsSync(path.join(root, "scripts", "start_local_app.sh")) &&
    fs.existsSync(path.join(root, "requirements.txt")) &&
    fs.existsSync(path.join(root, "app", "main.py"))
  );
}

function portableDataDir() {
  return process.env.GOLD_FREDICTOR_APP_DATA_DIR || path.join(app.getPath("userData"), "runtime");
}

function serviceConfig() {
  if (app.isPackaged) {
    if (process.env.GOLD_FREDICTOR_PORTABLE_MODE !== "1") {
      const projectCandidates = [
        process.env.GOLD_FREDICTOR_PROJECT_ROOT,
        path.resolve(process.resourcesPath, "../../../../.."),
        path.join(app.getPath("home"), "gold fredictor")
      ];
      const root = projectCandidates.find(isRunnableProjectRoot);
      if (root) {
        return { root, env: {} };
      }
    }

    const packagedRoot = path.join(process.resourcesPath, "app");
    if (hasPackagedSource(packagedRoot)) {
      const dataDir = portableDataDir();
      return {
        root: packagedRoot,
        env: {
          GOLD_FREDICTOR_BOOTSTRAP_VENV: "1",
          GOLD_FREDICTOR_APP_DATA_DIR: dataDir,
          GOLD_FREDICTOR_VENV_DIR: path.join(dataDir, ".venv"),
          GOLD_FREDICTOR_ENV_PATH: path.join(dataDir, ".env"),
          GOLD_FREDICTOR_DB_PATH: path.join(dataDir, "gold_monitor.db"),
          GOLD_FREDICTOR_LOG_DIR: path.join(dataDir, "logs"),
          GOLD_FREDICTOR_RUNTIME_DIR: path.join(dataDir, ".runtime")
        }
      };
    }
    throw new Error(
      "找不到可运行的项目目录或 App 内置源码。请重新安装 App，或设置 GOLD_FREDICTOR_PROJECT_ROOT。"
    );
  }
  return { root: path.resolve(__dirname, ".."), env: {} };
}

function waitFor(url, timeoutMs = 600000) {
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
  const { root, env } = serviceConfig();
  serviceRuntimeDir = env.GOLD_FREDICTOR_RUNTIME_DIR || path.join(root, ".runtime");
  const script = path.join(root, "scripts", "start_local_app.sh");
  serviceProcess = spawn("/bin/bash", [script, "--no-open"], {
    cwd: root,
    env: {
      ...process.env,
      ...env,
      GOLD_FREDICTOR_NO_OPEN: "1",
      GOLD_FREDICTOR_API_PORT: API_PORT,
      GOLD_FREDICTOR_DASHBOARD_PORT: DASHBOARD_PORT,
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

function stopServices() {
  if (serviceRuntimeDir) {
    for (const fileName of ["api.pid", "streamlit.pid"]) {
      const pidPath = path.join(serviceRuntimeDir, fileName);
      try {
        const pid = Number(fs.readFileSync(pidPath, "utf8").trim());
        if (pid > 0) {
          process.kill(pid, "SIGTERM");
        }
      } catch {
        // The service may have been started before this App instance or already exited.
      }
    }
  }
  if (serviceProcess && !serviceProcess.killed) {
    try {
      serviceProcess.kill("SIGTERM");
    } catch {
      // Ignore shutdown races.
    }
  }
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
    if (isAiChatUrl(url)) {
      createAiChatWindow(url);
      return { action: "deny" };
    }
    shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function isAiChatUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.origin === API_ORIGIN && parsed.pathname === "/ai/ui";
  } catch {
    return false;
  }
}

function createAiChatWindow(url = AI_CHAT_URL) {
  const chatWindow = new BrowserWindow({
    width: 1080,
    height: 760,
    minWidth: 860,
    minHeight: 620,
    title: "AI 对话",
    backgroundColor: "#0f172a",
    parent: mainWindow || undefined,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });

  childWindows.add(chatWindow);
  chatWindow.loadURL(url);
  chatWindow.webContents.setWindowOpenHandler(({ url: nextUrl }) => {
    if (isAiChatUrl(nextUrl)) {
      createAiChatWindow(nextUrl);
    } else {
      shell.openExternal(nextUrl);
    }
    return { action: "deny" };
  });
  chatWindow.on("closed", () => {
    childWindows.delete(chatWindow);
  });
}

async function boot() {
  try {
    startServices();
    await waitFor(API_HEALTH_URL, 600000);
    await waitFor(DASHBOARD_URL, 600000);
    createWindow();
  } catch (error) {
    dialog.showErrorBox(
      "Gold Fredictor 启动失败",
      `${error.message}\n\n本机项目模式请确认 .venv 已安装依赖，并可运行 ./scripts/start_local_app.sh。\n便携模式首次启动需要目标电脑已安装 python3，并允许联网安装依赖。`
    );
    app.quit();
  }
}

app.whenReady().then(boot);

app.on("before-quit", stopServices);

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
