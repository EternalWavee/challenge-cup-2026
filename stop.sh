#!/usr/bin/env bash
set -euo pipefail

stopped=0
for session in tianshu-chat tianshu-segformer; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux kill-session -t "${session}"
    echo "已停止：${session}"
    stopped=1
  fi
done

if [[ "${stopped}" == "0" ]]; then
  echo "已经成功停止所有服务"
fi
