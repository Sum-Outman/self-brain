# Contributing Guidelines

We welcome all forms of contribution! Thank you for your interest in the Self Brain project.

## 🌟 How to Contribute

### 1. Report Bugs 🐛
- Use [GitHub Issues](https://github.com/YOUR_USERNAME/self-brain/issues) to report problems
- Provide detailed error descriptions and reproduction steps
- Include system information and error logs

### 2. Feature Suggestions 💡
- Discuss new features in [GitHub Discussions](https://github.com/YOUR_USERNAME/self-brain/discussions)
- Submit feature requests using the Issue template
- Explain the purpose and expected behavior of the feature

### 3. Code Contributions 📝

#### Development Environment Setup
```bash
# 1. Fork the project
git clone https://github.com/YOUR_USERNAME/self-brain.git
cd self-brain

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Run tests
python -m pytest tests/
```

#### Code Guidelines
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) coding standards
- Use meaningful variable and function names
- Add necessary comments and docstrings
- Ensure code passes `flake8` and `black` checks

#### Commit Guidelines
- Use clear commit messages
- Follow [Conventional Commits](https://www.conventionalcommits.org/)
- Example format: `feat: add new training mode`

### 4. Documentation Improvements 📚
- Fix spelling errors and grammatical issues
- Add usage examples and tutorials
- Update API documentation
- Improve README files

## 🔄 Workflow

1. **Fork** the project to your GitHub account
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'feat: add amazing feature'`)
4. **Push your branch** (`git push origin feature/amazing-feature`)
5. **Create a Pull Request** to the main repository

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific tests
python -m pytest tests/test_training.py

# Run tests and generate coverage report
python -m pytest --cov=self_brain tests/
```

### Types of Tests
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test interactions between modules
- **End-to-End Tests**: Test complete system functionality

## 📋 Pull Request Template

### Title Format
```
[type]: [brief description]
```

### Content Template
```markdown
## 📋 Description
Briefly describe the changes in this pull request

## 🔗 Related Issues
Related Issue number: #123

## 🧪 Testing
- [ ] Added unit tests
- [ ] All existing tests pass
```