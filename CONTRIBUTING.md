"""Contribution guidelines."""

# Contributing to Battery Health Prediction

Thank you for your interest in contributing! Here's how you can help:

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install development dependencies: `pip install -e ".[dev]"`

## Development Workflow

### Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and use:

- **Black** for formatting: `black src/ tests/`
- **Flake8** for linting: `flake8 src/ tests/`
- **isort** for import sorting: `isort src/ tests/`
- **mypy** for type checking: `mypy src/`

### Writing Clean Code

Our codebase follows **Clean Code** principles:

✅ **DO**:

- Write functions that do one thing
- Use clear, intention-revealing names
- Keep functions small (5-20 lines)
- Add type hints
- Document edge cases and assumptions

❌ **DON'T**:

- Write overly complex functions
- Use ambiguous variable names
- Mix responsibilities in one function
- Skip error handling
- Leave TODO comments without issues

### Testing

- Write tests for new features
- Run: `pytest tests/ --cov=src`
- Aim for >80% code coverage

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add transformer model implementation
fix: handle edge case in sliding windows
docs: update architecture documentation
test: add tests for normalization
refactor: extract common logic to helpers
```

## Types of Contributions

### Bug Reports

Include:

- Minimum reproducible example
- Python and library versions
- Expected vs actual behavior

### Feature Requests

Describe:

- Problem being solved
- Proposed solution
- Expected use cases

### Code Contributions

1. Ensure code follows style guidelines
2. Add tests for new functionality
3. Update documentation
4. Submit pull request with clear description

## Pull Request Process

1. Update README if needed
2. Add/update tests
3. Ensure all tests pass
4. Run code quality checks
5. Submit PR with description

## Questions?

- Open a GitHub issue
- Check existing discussions
- Review documentation

Thank you for contributing! 🙏
