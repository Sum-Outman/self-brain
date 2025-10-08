# 🤝 Contributing to Self Brain

我们欢迎所有形式的贡献！感谢您对Self Brain项目的兴趣。

## 🌟 如何贡献

### 1. 报告Bug 🐛
- 使用 [GitHub Issues](https://github.com/YOUR_USERNAME/self-brain/issues) 报告问题
- 提供详细的错误描述和复现步骤
- 包含系统信息和错误日志

### 2. 功能建议 💡
- 在 [GitHub Discussions](https://github.com/YOUR_USERNAME/self-brain/discussions) 中讨论新功能
- 使用Issue模板提交功能请求
- 说明功能的用途和预期行为

### 3. 代码贡献 📝

#### 开发环境设置
```bash
# 1. Fork项目
git clone https://github.com/YOUR_USERNAME/self-brain.git
cd self-brain

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 3. 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. 运行测试
python -m pytest tests/
```

#### 代码规范
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 编码规范
- 使用有意义的变量名和函数名
- 添加必要的注释和文档字符串
- 确保代码通过 `flake8` 和 `black` 检查

#### 提交规范
- 使用清晰的提交消息
- 遵循 [Conventional Commits](https://www.conventionalcommits.org/)
- 示例格式: `feat: add new training mode`

### 4. 文档改进 📚
- 修复拼写错误和语法问题
- 添加使用示例和教程
- 更新API文档
- 改进README文件

## 🔄 工作流程

1. **Fork** 项目到您的GitHub账户
2. **创建功能分支** (`git checkout -b feature/amazing-feature`)
3. **提交更改** (`git commit -m 'feat: add amazing feature'`)
4. **推送分支** (`git push origin feature/amazing-feature`)
5. **创建Pull Request** 到主仓库

## 🧪 测试

### 运行测试
```bash
# 运行所有测试
python -m pytest

# 运行特定测试
python -m pytest tests/test_training.py

# 运行测试并生成覆盖率报告
python -m pytest --cov=self_brain tests/
```

### 测试类型
- **单元测试**: 测试单个函数和类
- **集成测试**: 测试模块间的交互
- **端到端测试**: 测试完整系统功能

## 📋 Pull Request 模板

### 标题格式
```
[type]: [brief description]
```

### 内容模板
```markdown
## 📋 描述
简要描述这次更改的内容

## 🔗 相关Issue
关联的Issue编号: #123

## 🧪 测试
- [ ] 添加了单元测试
- [ ] 所有现有测试通过
- [ ] 手动测试完成

## 📚 文档更新
- [ ] README已更新
- [ ] API文档已更新
- [ ] 添加/更新了代码注释

## 🎯 检查清单
- [ ] 代码遵循项目规范
- [ ] 自测通过
- [ ] 文档已更新
```

## 🎨 代码风格

### Python代码风格
- 使用4个空格缩进
- 最大行长度: 88字符
- 使用双引号字符串
- 函数名使用小写加下划线

### 命名规范
- **类名**: PascalCase (如: `TrainingManager`)
- **函数/变量**: snake_case (如: `start_training`)
- **常量**: UPPER_SNAKE_CASE (如: `MAX_MEMORY_MB`)

## 📞 联系方式

- **邮箱**: silencecrowtom@qq.com
- **GitHub Issues**: [项目Issues页面](https://github.com/YOUR_USERNAME/self-brain/issues)
- **讨论区**: [GitHub Discussions](https://github.com/YOUR_USERNAME/self-brain/discussions)

## 🙏 致谢

感谢所有为Self Brain项目做出贡献的开发者！您的贡献将帮助构建更好的AI系统。

## 📄 许可证

所有贡献都在 [Apache License 2.0](../LICENSE) 下发布。