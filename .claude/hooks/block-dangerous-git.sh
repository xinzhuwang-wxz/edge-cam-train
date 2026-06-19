#!/bin/bash
# Stage 0 护栏：拦截**不可逆/破坏性** git 命令，但**放行正常开发流**（普通 push/commit/checkout 分支等）。
# 设计原则：防 force-push / 删分支 / 丢未提交工作，不挡日常协作。
# 需要被拦的命令时，请用户在会话里用  ! <command>  亲自执行。

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

# 每个元素 = "正则@@人话说明"（用 @@ 分隔，因正则里含 | 字符）。只拦真正危险的子集。
DANGEROUS_PATTERNS=(
  "git push[^|;&]*--force@@强制推送 (--force) 会覆盖远端历史"
  "git push[^|;&]* -f($| )@@强制推送 (-f) 会覆盖远端历史"
  "git push[^|;&]*--delete@@经 push 删除远端分支"
  "git push[^|;&]* -d($| )@@经 push 删除远端分支"
  "git push[^|;&]+ :@@经 push 用 ':' refspec 删除远端分支"
  "git reset --hard@@reset --hard 丢弃未提交改动且不可恢复"
  "git clean [^|;&]*-[a-zA-Z]*f@@clean -f 永久删除未跟踪文件"
  "git branch [^|;&]*-D@@branch -D 强删未合并分支"
  "git checkout [^|;&]*-- ?\.($| )@@checkout -- . 丢弃工作区改动"
  "git checkout \.($| )@@checkout . 丢弃工作区改动"
  "git restore [^|;&]*\.($| )@@restore . 丢弃工作区改动"
)

for entry in "${DANGEROUS_PATTERNS[@]}"; do
  pattern="${entry%%@@*}"
  reason="${entry#*@@}"
  if echo "$COMMAND" | grep -qE "$pattern"; then
    echo "BLOCKED: '$COMMAND' — $reason。该命令被 Stage 0 护栏拦截；如确需执行，请用户用 '! <command>' 亲自运行。" >&2
    exit 2
  fi
done

exit 0
