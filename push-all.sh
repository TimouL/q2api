#!/bin/bash
# 双上游推送脚本
# 同时推送到GitHub和Codeberg仓库

echo "正在推送到所有上游仓库..."
git push origin main
echo "推送完成！"