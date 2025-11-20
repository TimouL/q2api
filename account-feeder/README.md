# Amazon Q 账号投喂服务

这是一个独立的账号投喂服务，用于让其他人通过 URL 登录投喂 Amazon Q 账号到主服务。

## 功能特点

- ✅ 极简单文件部署（仅 3 个核心文件）
- ✅ 无需数据库（使用内存存储临时会话）
- ✅ 自动设备授权流程（OIDC）
- ✅ 自动调用主服务创建账号
- ✅ 友好的 Web 界面

## 项目结构

```
amazonq-account-feeder/
├── app.py              # 单文件后端（FastAPI）
├── index.html          # 单文件前端（精简版）
├── requirements.txt    # Python 依赖
├── .env.example        # 配置示例
└── README.md           # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# 投喂服务端口（默认 8000）
PORT=8000

# 主服务地址（必须配置）
API_SERVER=http://localhost:3030
```

**重要：** 请确保 `API_SERVER` 指向正确的主服务地址！

### 3. 启动服务

```bash
python app.py
```

或使用 uvicorn：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. 访问服务

打开浏览器访问：`http://localhost:8000`

## 使用说明

### 投喂账号流程

1. 打开投喂服务页面（`http://localhost:8000`）
2. 填写账号标签（可选）
3. 点击"🚀 开始登录"按钮
4. 在自动打开的授权页面完成登录
5. 返回投喂页面，点击"⏳ 等待授权并创建账号"按钮
6. 等待最多 5 分钟，系统会自动创建账号并投喂到主服务
7. 看到"🎉 账号创建成功"提示即完成

### 注意事项

- ⏱️ 授权流程有 5 分钟超时限制
- 🔗 确保主服务正常运行且可访问
- 🌐 如果主服务在其他机器上，需要配置正确的 `API_SERVER` 地址
- 🔒 建议在内网环境使用，避免暴露到公网

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `PORT` | 投喂服务监听端口 | `8000` | 否 |
| `API_SERVER` | 主服务地址 | `http://localhost:3030` | 是 |
| `HTTP_PROXY` | HTTP 代理地址 | 无 | 否 |

### 主服务要求

主服务必须提供以下 API 端点：

- `POST /v2/accounts` - 创建账号接口

请求格式：

```json
{
  "label": "账号标签",
  "clientId": "客户端ID",
  "clientSecret": "客户端密钥",
  "refreshToken": "刷新令牌",
  "accessToken": "访问令牌",
  "enabled": true
}
```

## 技术架构

### 后端（app.py）

- **框架**: FastAPI
- **HTTP 客户端**: httpx
- **OIDC 授权**: 自实现（无第三方库依赖）
- **会话存储**: 内存字典（无需数据库）

### 前端（index.html）

- **技术栈**: 原生 HTML + CSS + JavaScript
- **样式**: 内联 CSS（深色主题）
- **交互**: 原生 Fetch API

### 工作流程

```
用户访问投喂页面
    ↓
点击"开始登录"
    ↓
后端注册 OIDC 客户端
    ↓
后端开始设备授权流程
    ↓
返回验证链接给用户
    ↓
用户在浏览器中完成授权
    ↓
点击"等待授权并创建账号"
    ↓
后端轮询获取 tokens（最多 5 分钟）
    ↓
后端调用主服务创建账号
    ↓
返回成功提示
```

## 常见问题

### Q: 为什么点击"等待授权并创建账号"后一直等待？

A: 可能的原因：
1. 未在授权页面完成登录
2. 授权已超时（5 分钟）
3. 网络连接问题

### Q: 创建账号失败怎么办？

A: 检查以下几点：
1. 主服务是否正常运行
2. `API_SERVER` 配置是否正确
3. 主服务的 `/v2/accounts` 接口是否可用
4. 查看主服务日志排查问题

### Q: 可以部署到公网吗？

A: 不建议直接暴露到公网，原因：
1. 无身份验证机制
2. 可能被滥用
3. 建议使用 VPN 或内网穿透

### Q: 如何限制投喂频率？

A: 当前版本未实现频率限制，可以考虑：
1. 在主服务层面限制
2. 使用 Nginx 限流
3. 添加验证码机制

## 部署建议

### 开发环境

```bash
python app.py
```

### 生产环境

使用 systemd 或 supervisor 管理进程：

```bash
# 使用 uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

### Docker 部署（可选）

创建 `Dockerfile`：

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py index.html ./
ENV PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]
```

构建并运行：

```bash
docker build -t amazonq-feeder .
docker run -d -p 8000:8000 \
  -e API_SERVER=http://host.docker.internal:3030 \
  amazonq-feeder
```

## 安全建议

1. ✅ 仅在可信网络环境使用
2. ✅ 定期清理内存中的过期会话
3. ✅ 使用 HTTPS（通过反向代理）
4. ✅ 添加访问日志监控
5. ✅ 限制投喂频率

## 许可证

与主项目保持一致

## 支持

如有问题，请联系主项目维护者或提交 Issue。
