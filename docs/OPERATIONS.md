# 运维与配置维护手册

本系统用于验证“零代码基础借助 AI 开发黄金预测系统”的可行性，默认启用
无人值守自我评估、自我修复、自我搜索、自我激活和自我进化。输出只用于
检验 AI 系统能否对标真实黄金走势，不用于黄金买卖参考。全自动不代表降低
准确性要求：自动激活仍必须通过样本量、命中率、方向准确率、MAPE、基线
提升和近期退化检查。

配置分为两类：

1. `.env`：密钥、数据库地址、外部服务地址、启动策略。修改后通常需要重启 API 或仪表盘。
2. `app_settings` 数据库表：自动驾驶与自我进化运行开关。可通过页面或 API 热更新，不需要改 `.env`。

## 统一管理入口

```bash
.venv/bin/python scripts/manage.py health
.venv/bin/python scripts/manage.py config --format table
.venv/bin/python scripts/manage.py self-heal --force
```

兼容旧入口仍可使用：

```bash
.venv/bin/python scripts/system_health_check.py
./scripts/start_local_app.sh
```

## 配置审计

API：

```text
GET /config/audit
```

命令行：

```bash
.venv/bin/python scripts/manage.py config
```

审计结果会标明：

- `source`：配置来自环境变量、默认值或数据库覆盖。
- `is_secret`：是否为敏感配置，输出会脱敏。
- `hot_reload`：是否可热更新。
- `requires_restart`：修改后是否需要重启进程。
- `status`：`ok`、`warn` 或 `empty`。

## 常用配置

| 配置 | 位置 | 说明 |
| --- | --- | --- |
| `DATABASE_URL` | `.env` | 数据库连接地址。 |
| `DASHBOARD_API_BASE_URL` | `.env` | Streamlit 调用 FastAPI 的地址。 |
| `FRED_API_KEY` | `.env` | FRED 官方 API Key。 |
| `NEWSAPI_KEY` | `.env` | NewsAPI Key；缺失时新闻能力降级。 |
| `FEISHU_WEBHOOK_URL` | `.env` | 飞书推送地址；为空时安全跳过。 |
| `DEEPSEEK_API_KEY` | `.env` | AI 助理 Key；为空时 AI 分析不可用。 |
| `AUTO_EVOLUTION_FULL_AUTO` | `app_settings`/页面 | AI 可行性验证全自动进化模式；默认开启，不需要用户确认，但不放宽质量门控。 |
| `AUTO_SELF_HEALING_ENABLED` | `app_settings`/页面 | 是否启用自动评估与修复。 |
| `AUTO_SELF_HEALING_AUTOFIX` | `app_settings`/页面 | 达标后是否自动修正/回滚。 |
| `AUTO_OPTIMIZE_*` | `app_settings`/页面 | 高级模型搜索和激活策略。 |

## 排障顺序

1. 先跑 `.venv/bin/python scripts/manage.py health`，确认系统是数据问题、预测闭环问题还是外部指标问题。
2. 再跑 `.venv/bin/python scripts/manage.py config --format table`，确认关键密钥和数据库覆盖是否符合预期。
3. 如果是自动化闭环问题，跑 `.venv/bin/python scripts/manage.py self-heal --force`。
4. 修改 `.env` 后重启；修改页面自动驾驶开关后无需重启。

系统只提供 AI 可行性验证所需的数据分析、黄金多空评分和风险提示，不用于黄金买卖参考。
