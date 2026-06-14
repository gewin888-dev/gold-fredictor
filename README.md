# 黄金走势实时监控与预测系统

一个 Python MVP，用于采集黄金价格、FRED 宏观数据、CFTC 持仓、中国黄金溢价、央行购金、新闻情绪和宏观事件，并通过规则评分、回测、预测、Streamlit 仪表盘和飞书告警做监控。系统只提供数据分析、评分和风险提示，不直接给出买入、卖出、加仓、减仓等投资建议。

## 功能

- FRED 官方 API 数据采集
- CFTC 官方 COT 持仓数据采集
- Yahoo Finance 金价采集
- 中国黄金溢价、央行购金、NewsAPI/GDELT 新闻情绪采集
- SQLite + SQLAlchemy 数据库存储
- 黄金多空评分计算
- 评分回测
- 金价预测
- 参数搜索与评分参数版本管理
- FastAPI 接口
- 数据健康检查
- 宏观事件日历
- APScheduler 定时任务预留
- Streamlit 仪表盘
- 飞书机器人推送接口预留
- 飞书评分告警

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填入 `FRED_API_KEY` 和 `NEWSAPI_KEY`。如果暂时没有飞书机器人，可保持 `FEISHU_WEBHOOK_URL` 为空。

默认情况下，API 启动只建表，不会自动联网采集或启动后台调度器。需要自动启动时可在 `.env` 设置：

```env
AUTO_START_SCHEDULER=true
AUTO_BOOTSTRAP_DATA=true
PRODUCTION_MODE=true
PREDICTION_SCORE_SOURCES=backfill_real_v2,rule_v2
NEWSAPI_DAILY_LIMIT=100
```

初始化数据库：

```bash
python scripts/init_db.py
```

如果暂时没有 `FRED_API_KEY`，可以加载本地样例数据，先体验完整闭环：

```bash
python scripts/load_sample_data.py
python scripts/load_sample_events.py
python scripts/compute_score.py
streamlit run dashboard/streamlit_app.py
```

采集 FRED 数据：

```bash
python scripts/collect_fred.py
```

采集 CFTC 黄金期货持仓：

```bash
python scripts/collect_cftc.py
```

计算黄金多空评分：

```bash
python scripts/compute_score.py
```

推送最新评分告警到飞书：

```bash
python scripts/send_score_alert.py
```

启动仪表盘：

```bash
streamlit run dashboard/streamlit_app.py
```

可选启动 API：

```bash
uvicorn app.main:app --reload
```

访问 API 文档：

```text
http://127.0.0.1:8000/docs
```

运行测试：

```bash
pytest -q
```

可选启动每日定时采集与评分任务：

```bash
python scripts/run_scheduler.py
```

该脚本以前台方式运行，按 `Ctrl+C` 可停止。当前默认注册小时级采集评分任务和每日完整报告任务。

macOS 可以安装用户级 `launchd` 后台任务，让调度器在登录后自动运行：

```bash
scripts/install_macos_launchd.sh
```

日志写入 `logs/launchd.out.log` 和 `logs/launchd.err.log`。自我优化默认关闭；只有设置 `AUTO_OPTIMIZE_SCORE_PARAMS=true` 才会每周生成候选参数，只有同时设置 `AUTO_ACTIVATE_OPTIMIZED_PARAMS=true` 且命中率超过 `AUTO_OPTIMIZE_MIN_HIT_RATE` 才会自动激活。

## API

- `GET /health`
- `GET /health/data`
- `GET /gold/price`
- `GET /predict/gold`
- `POST /collect/fred`
- `POST /collect/cftc`
- `POST /collect/gold_history`
- `POST /collect/china_premium`
- `POST /collect/cb_gold`
- `POST /collect/sentiment`
- `POST /score/compute`
- `POST /score/optimize`
- `GET /score/params`
- `GET /backtest/score?horizon_days=20`
- `GET /macro/latest`
- `GET /china_premium/latest`
- `GET /cb_gold/latest`
- `GET /sentiment/latest`
- `GET /events/upcoming?days_ahead=30`
- `GET /positions/cftc/latest`
- `GET /score/latest`
- `POST /notify/feishu/test`

## 第一批 FRED 指标

| series_id | 含义 |
| --- | --- |
| `DGS10` | 美国 10 年期国债收益率 |
| `DFII10` | 美国 10 年期 TIPS 实际收益率 |
| `T10YIE` | 10 年通胀预期 |
| `FEDFUNDS` | 联邦基金利率 |
| `VIXCLS` | VIX 恐慌指数 |
| `DTWEXBGS` | 美元广义贸易加权指数 |

金价不再从 FRED 读取，当前 MVP 使用新浪财经 COMEX 黄金期货 `hf_GC` 写入 `gold_prices` 和日内快照表；历史兼容路径可保留 Yahoo 语义。新浪/Yahoo 都属于免费或延迟行情源，适合 MVP 展示和开发验证，不应视为交易级实时行情。

## 评分说明

评分范围为 `-100` 到 `100`：

- `>= 30`：偏多
- `-30 ~ 30`：中性
- `<= -30`：偏空

评分由短期动量、中期宏观和长期宏观三类因子聚合而成。当前已纳入实际利率、实际利率曲线、名义利率、联邦基金利率、美元指数、VIX、通胀预期、期限溢价、美元流动性、财政压力、黄金趋势、短期动量、CFTC 持仓、中国溢价、央行购金、新闻情绪、ETF 资金流、期权隐波代理，以及白银、铜金比、GDX、WTI 等辅助结构指标。该评分仅用于数据分析和风险提示。

如果数据库中存在可信且未过期的 CFTC 黄金期货持仓，评分会额外加入 `CFTC投机仓位` 因子。中国黄金溢价、央行购金、新闻情绪和外部市场结构指标也有因子接口，但 `SAMPLE`、`ESTIMATE`、`MANUAL`、`JSON` 等样本或占位来源默认不纳入评分，只在风险提示中说明。生产模式下页面默认隐藏或弱化这些低可信来源。

COMEX 库存、COMEX 期限结构、地缘风险和中印实物需求当前可在页面灰色展示，不参与自动评分；后续接入稳定数据源或手动录入后可逐步纳入。

## 数据质量分层

系统会对每条数据的 `source` 做质量分层，并在 `GET /health/data` 和 Streamlit 数据健康表中展示：

- `官方/授权源`：如 `FRED`、`CFTC`、`WGC`、`IMF`、`SGE`、`LBMA`。
- `免费/延迟源`：如 `SINA`、`YAHOO`、`NEWSAPI`、`GDELT`。
- `样本/占位源`：如 `SAMPLE`、`ESTIMATE`、`MANUAL`、`JSON`，默认不作为可信评分源。
- `模型派生`：如 `rule_v1`、`optimized_active`、`auto_trigger`。

如果核心宏观或金价输入来自样本/占位源，系统仍允许本地演示计算，但会在评分摘要和风险提示中标明“不能视为生产级评分”。

## CFTC 数据

MVP 使用 CFTC 官方当前周报文件：

```text
https://www.cftc.gov/dea/newcot/deafut.txt
```

当前只抽取 `GOLD - COMMODITY EXCHANGE INC.`，合约市场代码 `088691`。采集字段包括总持仓、非商业多头、非商业空头、非商业套利、商业多头、商业空头，并计算非商业净持仓。

## 数据健康检查

`GET /health/data` 会检查 FRED 指标、CFTC 黄金持仓、中国黄金溢价、央行购金、新闻情绪和最新评分是否缺失或过期，并返回整体状态：

- `ok`：数据新鲜度正常。
- `warn`：部分数据偏旧，需要关注。
- `error`：关键数据缺失或明显过期。

Streamlit 首页也会展示同样的数据健康表。

## 新闻情绪

生产新闻源保留 NewsAPI，并兼容 GDELT。`NEWSAPI_KEY` 必须放在 `.env`，源码不保存密钥。`NEWSAPI_DAILY_LIMIT` 用于限制单次采集规模，避免超过免费配额。采集前会按 `source_url` 查询数据库，已存在的新闻不会重复入库。

## 央行购金

央行购金数据为手动维护的 WGC 来源数据，存放在 `data/cb_gold_monthly.json`。更新该 JSON 后调用 `POST /collect/cb_gold` 或定时任务即可刷新数据库。

## 评分回测

`GET /backtest/score?horizon_days=20` 会把历史评分与黄金价格对齐，计算指定未来周期的收益表现。API 默认只返回 summary；如需交易明细，可加分页参数：

```text
GET /backtest/score?horizon_days=20&include_trades=true&limit=100&offset=0
```

当前生产训练与回测默认只使用 `.env` 中 `PREDICTION_SCORE_SOURCES` 指定的同版本评分源，例如 `backfill_real_v2,rule_v2`，避免混合 `rule_v1`、`optimized_active` 等不同口径。

summary 包括：

- 样本数
- 方向样本数
- 方向命中率
- 平均未来收益
- 多头、空头、中性样本数量

该回测只用于评估评分规则的历史表现，不构成交易建议。

## 自我进化与模型健康

系统支持受控自我进化：自动搜索候选评分参数和候选预测模型，但正式交付默认不自动激活。仪表盘中的“模型健康”会展示：

- 当前评分参数版本：默认规则或激活版本。
- 当前预测模型版本。
- 最近评分候选和预测候选。
- 已评估预测数、到期未评估预测数。
- 当前样本是否满足自动优化条件。
- 自动搜索 / 自动激活开关状态。
- 不能自我进化的具体原因。

相关 API：

```text
GET  /settings/auto-optimize
POST /settings/auto-optimize
POST /score/optimize
GET  /score/params
POST /score/params/{version}/activate
POST /score/params/deactivate
POST /predict/models/optimize
POST /predict/models/{version}/activate
```

`POST /settings/auto-optimize` 只允许修改自动优化开关，并写入数据库 `app_settings` 表，不会改写 `.env` 或任何密钥字段。详细治理说明见 `docs/SELF_EVOLUTION_GOVERNANCE.md`。

## 飞书告警

`scripts/send_score_alert.py` 会读取最新评分、主要因子、风险提示和数据健康状态，并推送到飞书机器人。

如果存在未来宏观事件，告警也会附带最近事件摘要。如果 `.env` 未配置 `FEISHU_WEBHOOK_URL`，脚本会安全跳过，不会报错中断。每日定时任务也会在采集和评分后尝试发送同样的告警。

## 宏观事件日历

`GET /events/upcoming?days_ahead=30` 会返回未来宏观事件。当前 MVP 使用本地可维护的事件表，适合记录 CPI、PCE、FOMC、非农等关键事件。

加载样例事件：

```bash
python scripts/load_sample_events.py
```

## 常见问题

- 如果运行 `python` 提示命令不存在，请使用 `python3`，或激活虚拟环境后使用 `.venv/bin/python`。
- 如果 `scripts/collect_fred.py` 提示缺少 `FRED_API_KEY`，请先在 `.env` 填入 FRED API Key，或先运行 `scripts/load_sample_data.py` 演示本地闭环。
- 如果没有配置飞书机器人，飞书推送接口会返回跳过状态，不会中断系统运行。
