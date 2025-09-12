# 分支保护规则设置指南

## 如何设置只有你能修改仓库

### 步骤1：访问分支保护设置
访问：https://github.com/Sum-Outman/self-brain/settings/branches

### 步骤2：添加分支保护规则
1. 点击 "Add branch protection rule"
2. Branch name pattern: 输入 `main`
3. 勾选以下选项：
   - ✅ Restrict pushes that create files larger than 100MB
   - ✅ Require a pull request before merging
   - ❌ 不勾选 "Allow force pushes"
   - ❌ 不勾选 "Allow deletions"
   - ✅ Include administrators

### 步骤3：验证设置
设置完成后，只有你能：
- 直接push到main分支
- 合并pull requests
- 管理仓库设置

其他人只能：
- 查看代码
- fork项目（创建副本）
- 提交issues和pull requests