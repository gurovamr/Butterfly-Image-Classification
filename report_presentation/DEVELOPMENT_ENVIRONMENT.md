# Development Environment Setup

This document describes the development environment configuration for the Butterfly Image Classification project, covering the five key aspects required for an effective development workflow.

---

## 1. Choose IDE

**Selected IDE: Visual Studio Code (VS Code)**

### Why VS Code?
- Lightweight and fast
- Excellent Python extension ecosystem
- Built-in Git integration
- Integrated terminal support
- Wide community adoption

### Configuration File
- **Location:** `.vscode/settings.json` and `.vscode/launch.json`

### IDE Extensions Used
- **Python** (ms-python.python) - Official Python language support
- **Pylance** (ms-python.vscode-pylance) - Python type checking and IntelliSense
- **Ruff** (charliermarsh.ruff) - Fast linter and formatter
- **Pytest** (littlefineprint.pytest) - Test discovery and execution
- **GitHub Pull Requests and Issues** (GitHub.vscode-pull-request-github) - GitHub integration

---

## 2. Configure Python/Toolchain

### Python Version
- **Minimum:** Python 3.10
- **Tested on:** Python 3.10, 3.11, 3.12 (via GitHub Actions matrix testing)

### Environment Management

#### Option 1: Virtual Environment (venv) - **Recommended for development**
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

#### Option 2: Conda Environment
```powershell
# Run the provided setup script (Windows)
.\scripts\setup_conda.ps1

# Or manually create conda environment
conda create -n butterfly-classification python=3.10
conda activate butterfly-classification
```

### Dependency Management

#### Installation Locations
- **Project configuration:** `pyproject.toml`
- **Core dependencies:** Defined in `[project] dependencies`
- **Development extras:** `[project.optional-dependencies]`
  - `dev` group: linting/formatting tools (ruff, black, pylint)
  - `test` group: testing tools (pytest, pytest-mock, pytest-cov, coverage)

#### Installation Commands

```bash
# Install runtime dependencies only
pip install .

# Install with development tools (linters, formatters)
pip install ".[dev]"

# Install with testing tools
pip install ".[test]"

# Install all (dev + test) - **Recommended for development**
pip install ".[dev,test]"

# Install in editable mode with all extras (best for active development)
pip install -e ".[dev,test]"
```

### Package Distribution
- **Build system:** setuptools
- **Entrypoint:** `butterfly-classifier` command-line script pointing to `main.py`
- **Installed packages declared in:** `pyproject.toml [tool.setuptools]`

### Key Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥2.2.6 | Numerical computing |
| pandas | ≥2.3.3 | Data manipulation |
| scikit-learn | ≥1.7.2 | ML utilities |
| matplotlib | ≥3.10.8 | Plotting |
| seaborn | ≥0.13.2 | Statistical visualization |
| tensorflow | ≥2.21.0 | Deep learning framework |
| pillow | ≥12.1.1 | Image processing |

---

## 3. Add Linters/Formatters

### Tools Configured

#### Ruff
**Purpose:** Fast, unified linter and code formatter (modern replacement for flake8 + isort)

**Configuration** (in `pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
target-version = "py310"
```

**Usage:**
```bash
# Check for linting issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .
```

#### Black
**Purpose:** Opinionated Python code formatter (ensures consistent style)

**Configuration** (in `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ["py310"]
```

**Usage:**
```bash
# Format all files
black .

# Check formatting without modifying
black --check .
```

#### Pylint
**Purpose:** Code quality analyzer (catches potential bugs and style issues)

**Configuration** (in `pyproject.toml`):
```toml
[tool.pylint.master]
extension-pkg-allow-list = ""
ignored-modules = ["tensorflow", "tensorflow.keras", "keras"]

[tool.pylint.design]
max-args = 8
max-locals = 25

[tool.pylint.format]
max-line-length = 100
```

**Usage:**
```bash
# Run pylint on specific file or directory
pylint scripts/
```

### VS Code Integration

**File:** `.vscode/settings.json`

```json
{
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true
    },
    "ruff.enable": true,
    "ruff.lint.enable": true
}
```

**Result:** Code is automatically formatted on save using Ruff as the default formatter.

### CI/CD Linting
Linting runs automatically in GitHub Actions (`.github/workflows/ci.yml`):
- **Ruff check:** `ruff check .`
- **Black format check:** `black --check .`

---

## 4. Add Debugger/Testing Integration

### Testing Framework: Pytest

**Configuration** (in `pyproject.toml`):
```toml
[tool.pytest.ini_options]
addopts = "--doctest-modules --cov=scripts --cov-fail-under=80"
testpaths = ["tests", "scripts"]
```

#### Key Testing Features

1. **Unit Tests**
   - Location: `tests/` directory
   - Files: `test_*.py` (pytest auto-discovery)
   - Includes fixtures defined in `tests/conftest.py`

2. **Doctest Support**
   - Inline tests in docstrings are executed
   - Added via `--doctest-modules` in pytest options

3. **Code Coverage**
   - Tool: `pytest-cov` with `coverage` library
   - Target: 80% minimum coverage (enforced: `--cov-fail-under=80`)
   - Coverage configuration in `pyproject.toml`:
     ```toml
     [tool.coverage.run]
     source = ["scripts"]
     omit = ["*/tests/*", "*/__init__.py"]
     
     [tool.coverage.report]
     show_missing = true
     exclude_lines = ["if __name__ == .__main__.:"]
     ```

4. **Test Fixtures** (in `tests/conftest.py`)
   - `make_image()`: Factory fixture for generating fake image arrays
   - Useful for mocking data in tests

5. **Mocking Support**
   - Tool: `pytest-mock`
   - Provides `mocker` fixture for patching

### Debugger Configuration

**File:** `.vscode/launch.json`

Two debug configurations are provided:

#### Configuration 1: Debug main.py
```json
{
    "name": "Python: main.py",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/main.py"
}
```

#### Configuration 2: Debug Pytest
```json
{
    "name": "Python: Pytest",
    "type": "debugpy",
    "request": "launch",
    "module": "pytest",
    "args": ["tests", "-v"],
    "justMyCode": false
}
```

**How to use:**
1. Set breakpoints in VS Code (click on line number)
2. Press **F5** or go to Run → Start Debugging
3. Select the configuration to run
4. Debugger will pause at breakpoints

### IDE Test Integration

**File:** `.vscode/settings.json`

```json
{
    "python.testing.pytestArgs": ["tests"],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true
}
```

**Result:**
- VS Code's Test Explorer shows all tests
- Run/debug individual tests or entire suites from the UI
- Green checkmarks indicate passing tests
- Red X marks failing tests

### CI/CD Test Pipeline

**File:** `.github/workflows/ci.yml`

```yaml
test:
  runs-on: ubuntu-latest
  needs: lint
  steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10.14'
        cache: 'pip'
    - name: Install dependencies
      run: pip install ".[dev,test]"
    - name: Run tests + doctests + coverage
      run: pytest --doctest-modules --cov=scripts --cov-report=xml --cov-fail-under=80
    - name: Upload coverage
      uses: codecov/codecov-action@v4
```

**Additional CI Workflow:** `.github/workflows/matrix.yml`
- Tests on Python 3.10, 3.11, and 3.12
- Includes pylint analysis
- Ensures compatibility across Python versions

---

## 5. Make Tests Easy to Run Locally

### Running Tests Locally

#### Basic Test Execution
```bash
# Run all tests
pytest

# Run tests in verbose mode
pytest -v

# Run tests with live output
pytest -s
```

#### With Coverage Report
```bash
# Show coverage percentage
pytest --cov=scripts

# Show missing lines
pytest --cov=scripts --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=scripts --cov-report=html
# Open htmlcov/index.html in browser
```

#### Run Specific Tests
```bash
# Run single test file
pytest tests/test_dataset.py

# Run single test function
pytest tests/test_dataset.py::test_normalize

# Run tests matching pattern
pytest -k "download" -v
```

#### Include Doctests
```bash
# Run doctests along with unit tests
pytest --doctest-modules
```

### pytest Configuration Benefits

1. **Auto-Discovery**
   - Automatically finds tests in `tests/` directory
   - Discovers functions matching `test_*.py` pattern
   - No manual configuration needed

2. **Fixture Support**
   - Shared fixtures in `conftest.py` available to all tests
   - Example: `make_image()` factory used across multiple test files

3. **Coverage Enforcement**
   - Minimum 80% coverage required
   - Tests fail if coverage drops below threshold
   - Prevents regressions in code quality

4. **Output Control**
   - `-v` flag shows each test result
   - `-s` flag prints stdout/stderr
   - `-x` flag stops at first failure

### One-Command Setup

After cloning the repository:
```bash
# Clone repo
git clone https://github.com/gurovamr/Butterfly-Image-Classification.git
cd Butterfly-Image-Classification

# Create environment and install
python -m venv .venv
.venv\Scripts\activate          # Windows
# or: source .venv/bin/activate # macOS/Linux

# Install all dependencies (dev + test)
pip install ".[dev,test]"

# Run tests immediately
pytest
```

### Quick Reference Commands

| Command | Purpose |
|---------|---------|
| `pytest` | Run all tests |
| `pytest -v` | Verbose output |
| `pytest -s` | Show print statements |
| `pytest -x` | Stop at first failure |
| `pytest --cov=scripts` | Run with coverage |
| `pytest --cov-report=html` | Generate HTML coverage report |
| `pytest tests/test_dataset.py` | Run single test file |
| `pytest -k keyword` | Run tests matching keyword |
| `ruff check .` | Check code style |
| `black .` | Format code |
| `black --check .` | Check formatting |

### VS Code Test Explorer

1. Open the Test Explorer in VS Code (left sidebar)
2. Tests appear with hierarchical structure
3. Run tests by clicking:
   - Green triangle → Run test
   - Debug icon → Debug test
   - Refresh icon → Discover tests

---

## Summary

| Aspect | Solution | Status |
|--------|----------|--------|
| **IDE** | VS Code with Python extension | ✅ Configured |
| **Python/Toolchain** | Python 3.10+, venv/conda, pip+setuptools | ✅ Configured |
| **Linters/Formatters** | Ruff + Black + Pylint | ✅ Configured + CI/CD integrated |
| **Debugger/Testing** | pytest + debugpy + coverage (80% minimum) | ✅ Configured |
| **Easy Test Execution** | Single `pytest` command, VS Code integration | ✅ Configured |

All development environment components are production-ready and integrated with CI/CD workflows on GitHub Actions.
