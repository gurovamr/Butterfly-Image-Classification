# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment workflows implemented in the Butterfly Image Classification project using GitHub Actions.

---

## Part I: Continuous Integration (CI)

The CI pipeline automatically validates code quality, runs tests, and reports coverage on every push and pull request.

### 1. GitHub Actions Workflow Overview

**File:** `.github/workflows/ci.yml`

#### Workflow Triggers

```yaml
on:
  push:
    branches: [main]      # Runs when code is pushed to main
  pull_request:           # Runs on every pull request (any branch)
```

**Result:** Code is checked before merging to `main`. Every PR gets automatic validation.

#### Workflow Structure

The CI pipeline has **two sequential jobs:**

```
┌─────────────────────────────────────────────────────────┐
│ 1. LINT JOB (runs first)                                 │
│    - Check code style with Ruff                          │
│    - Check formatting with Black                         │
│    - Takes ~1-2 minutes                                  │
└─────────────────────────────────────────────────────────┘
                           ↓
                    (Only if LINT passes)
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 2. TEST JOB (runs second)                                │
│    - Run unit tests + doctests                           │
│    - Measure coverage                                    │
│    - Upload coverage to Codecov                          │
│    - Publish results to PR                               │
│    - Takes ~5-10 minutes                                 │
└─────────────────────────────────────────────────────────┘
```

**Key:** If linting fails, tests don't run (fail fast principle).

---

### 2. Formatter/Linter Checks

**Job:** `lint` (runs on `ubuntu-latest`)

#### Step 1: Environment Setup
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.10.14'
    cache: 'pip'
```
- Installs Python 3.10.14
- Caches pip packages for faster runs
- Reuses cache if dependencies haven't changed

#### Step 2: Install Dev Tools
```yaml
- name: Install dev dependencies
  run: pip install ".[dev]"
```
Installs: `ruff`, `black`, `pylint` (defined in `pyproject.toml [project.optional-dependencies] dev`)

#### Step 3: Ruff Check (Fast Linting)
```yaml
- name: Lint
  run: ruff check .
```

**What it checks:**
- Unused imports
- Undefined variables
- Syntax errors
- PEP 8 violations
- Circular imports
- Duplicate code

**Fail condition:** Any violation fails the job

#### Step 4: Black Check (Format Validation)
```yaml
- name: Format check
  run: black --check .
```

**What it checks:**
- Line length (max 100)
- Spacing consistency
- Indentation
- String quote style

**Fail condition:** Any file not matching Black's style fails the job

### What Happens on Failure?

If either check fails:
1. ❌ Red X appears on PR
2. Tests don't run (blocked by `needs: lint`)
3. PR cannot be merged until fixed
4. Developer fixes code locally: `black . && ruff check . --fix`

---

### 3. Tests on PRs and Merges

**Job:** `test` (runs only if `lint` succeeds)

#### Environment Setup
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.10.14'
    cache: 'pip'
```

#### Install All Dependencies
```yaml
- name: Install dependencies
  run: pip install ".[dev,test]"
```

Installs:
- Runtime dependencies: numpy, pandas, tensorflow, etc.
- Dev tools: ruff, black, pylint
- Test tools: pytest, pytest-cov, pytest-mock, coverage

#### Run Tests with Coverage
```yaml
- name: Run tests + doctests + coverage
  run: pytest --doctest-modules --cov=scripts --cov-report=xml --cov-fail-under=80 --junitxml=junit/test-results.xml
```

**What this command does:**
| Flag | Meaning |
|------|---------|
| `--doctest-modules` | Also run tests in docstrings |
| `--cov=scripts` | Measure coverage only for `scripts/` module |
| `--cov-report=xml` | Generate XML report for Codecov |
| `--cov-fail-under=80` | **FAIL if coverage < 80%** |
| `--junitxml=...` | Generate test results XML for GitHub |

**Example failure message:**
```
FAILED: coverage is 75.2%, target is 80%
```

**Test files run:**
- `tests/test_*.py` — All unit tests
- `scripts/*.py` — Doctests in function docstrings
- Example: `scripts/model.py` docstrings are executed as tests

#### Upload Test Results as Artifacts
```yaml
- name: Upload test results artifact
  uses: actions/upload-artifact@v4
  with:
    name: pytest-results
    path: junit/test-results.xml
  if: always()
```

Creates downloadable XML file with test details. Runs even if tests fail (`if: always()`).

#### Publish Results to PR
```yaml
- name: Publish test results to PR
  uses: EnricoMi/publish-unit-test-result-action@v2
  if: "!cancelled()"
  with:
    files: junit/*.xml
```

**Result in PR:** 
```
✅ All tests passed (145 tests)
✅ Coverage: 82.1%
```

Or on failure:
```
❌ 3 tests failed
   - test_normalize (FAILED)
   - test_augmentation (ERROR)
✅ Coverage: 75.2% (below 80% target)
```

---

### 4. Coverage Reporting (Codecov)

**File:** `codecov.yml`

#### Coverage Thresholds
```yaml
coverage:
  status:
    project:
      default:
        target: 80%           # Overall coverage must be ≥80%
        threshold: 1%         # Allow 1% drop from main branch
    patch:
      default:
        target: 80%           # New code must be ≥80% covered
        threshold: 1%         # Allow 1% lower than main
```

#### Upload to Codecov (in ci.yml)
```yaml
- name: Upload coverage
  uses: codecov/codecov-action@v4
  with:
    token: ${{ secrets.CODECOV_TOKEN }}  # Secret stored in GitHub
    files: coverage.xml
    fail_ci_if_error: false              # Don't fail if upload fails
```

#### What Happens After Upload?

1. **Codecov.io Dashboard:** Coverage reports visible at https://codecov.io/gh/gurovamr/Butterfly-Image-Classification
2. **PR Comment:** Codecov bot comments on PRs showing:
   - Overall coverage % change
   - Uncovered lines
   - File-by-file breakdown

**Example PR comment from Codecov:**
```
📊 Coverage report: 82.1%
  ✅ Overall coverage improved from 80.2% to 82.1%
  📈 +1.9% change

  Files Changed:
  - scripts/data_preprocessing.py: 85% → 90% (+5%)
  - scripts/model.py: 75% → 78% (+3%)
```

#### Badge in README
The project displays a coverage badge:
```markdown
[![codecov](https://codecov.io/gh/gurovamr/Butterfly-Image-Classification/branch/main/graph/badge.svg)](https://codecov.io/gh/gurovamr/Butterfly-Image-Classification)
```

---

### 5. Matrix Testing (Optional but Implemented)

**File:** `.github/workflows/matrix.yml`

Tests the project across multiple Python versions to ensure compatibility.

#### Test Matrix Configuration
```yaml
strategy:
  fail-fast: false          # Don't stop if one version fails
  matrix:
    python-version: ["3.10.14", "3.11.9", "3.12.3"]
```

#### Parallel Test Runs

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Python 3.10.14   │  │ Python 3.11.9    │  │ Python 3.12.3    │
│                  │  │                  │  │                  │
│ Run tests        │  │ Run tests        │  │ Run tests        │
│ (5-10 min)       │  │ (5-10 min)       │  │ (5-10 min)       │
└──────────────────┘  └──────────────────┘  └──────────────────┘
       ✅ PASS               ✅ PASS              ✅ PASS
```

**Result:** All three versions run in parallel, saving time.

#### Test Command
```yaml
- name: Run tests
  run: pytest --tb=short -q
```
- `-q`: Quiet output (one line per test file)
- `--tb=short`: Short traceback on failures

#### Pylint Code Quality Check
```yaml
pylint:
  runs-on: ubuntu-latest
  steps:
    - name: Run pylint
      run: pylint scripts/ main.py --fail-under=8.0
```

**What pylint checks:**
- Code complexity
- Design issues
- Function argument counts
- Naming conventions

**Fail condition:** Pylint score < 8.0 fails the workflow

#### Matrix Workflow Triggers
```yaml
on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:          # Manual trigger from GitHub UI
```

---

## Part II: Continuous Deployment (CD) & Release

### 6. Release Workflow

**File:** `.github/workflows/build_binaries.yml`

Automatically builds and releases executable binaries when a version tag is pushed.

#### Workflow Trigger
```yaml
on:
  push:
    tags:
      - 'v*'                 # Triggered on tags like v0.1.0, v1.0.0
  workflow_dispatch:         # Manual trigger from GitHub UI
```

#### Release Process

```
Developer pushes v0.1.0 tag
        ↓
GitHub Actions triggered
        ↓
Build for 3 platforms (Linux, Windows, macOS) in parallel
        ↓
Create GitHub Release
        ↓
Upload binaries + checksums + release notes
```

---

### 7. Packaging with PyInstaller

**What it does:** Converts Python code into standalone executables (no Python installation needed)

#### Build Matrix
```yaml
strategy:
  fail-fast: false
  matrix:
    include:
      - os: ubuntu-latest
        target: linux
        extension: ""
      - os: windows-latest
        target: windows
        extension: ".exe"
      - os: macos-latest
        target: macos
        extension: ""
```

**Result:** Creates 3 executables in parallel:
- `butterfly-classifier-linux` (Linux)
- `butterfly-classifier-windows.exe` (Windows)
- `butterfly-classifier-macos` (macOS)

#### Build Steps

**Step 1: Install Dependencies**
```yaml
- name: Install dependencies
  run: |
    pip install .
    pip install pyinstaller
```

**Step 2: Build Executable**
```bash
pyinstaller \
  --onefile \                              # Single executable file
  --name "butterfly-classifier-${target}" \
  --hidden-import=tensorflow \             # Include TensorFlow
  --hidden-import=sklearn.utils._cython_blas \
  --collect-all tensorflow \               # Bundle TensorFlow data
  main.py
```

**Key options:**
| Option | What it does |
|--------|-------------|
| `--onefile` | Bundle everything into one .exe file |
| `--hidden-import=tensorflow` | Include TensorFlow (not auto-detected) |
| `--collect-all tensorflow` | Include TensorFlow data files |
| `main.py` | Entry point |

#### Generate SHA256 Checksums
```bash
python - <<'PY'
import hashlib
from pathlib import Path
p = Path("release") / output_name
sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
(Path("release") / f"{output_name}.sha256").write_text(sha256)
print(f"SHA256: {sha256}")
PY
```

**Why checksums?** Users can verify the file wasn't corrupted or tampered with:
```bash
# User verifies on their machine
sha256sum butterfly-classifier-linux > computed.sha256
diff computed.sha256 butterfly-classifier-linux.sha256
```

---

### 8. Create GitHub Release

**Step: Create or Update Release**
```yaml
- name: Create or update GitHub Release
  uses: softprops/action-gh-release@v2
  with:
    files: release/*                  # Upload all files in release/
    generate_release_notes: true       # Auto-generate changelog
```

#### What Appears in GitHub Release

After tagging `v0.1.0`:

```
📦 v0.1.0
Release published • Commits since last release: 42

Assets:
 📥 butterfly-classifier-linux (125 MB)
 📥 butterfly-classifier-linux.sha256
 📥 butterfly-classifier-windows.exe (140 MB)
 📥 butterfly-classifier-windows.exe.sha256
 📥 butterfly-classifier-macos (118 MB)
 📥 butterfly-classifier-macos.sha256

Release Notes (auto-generated):
✨ Features:
  - Add image augmentation pipeline
  - Improve CNN architecture with batch norm

🐛 Bug Fixes:
  - Fix validation split calculation
  - Handle missing images gracefully
```

**Users can download binaries directly** without needing Python installed.

---

### 9. Optional: PyPI/TestPyPI Publishing

**Current Status:** ❌ NOT implemented in this project

**Why not?**
- This is an **application** (runnable pipeline), not a reusable library
- Executable binaries are sufficient for distribution
- Users don't need to import this as a Python package

**If it were a library, here's how you'd do it:**

```yaml
- name: Build distributions
  run: |
    pip install build
    python -m build

- name: Publish to TestPyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: https://test.pypi.org/legacy/
    password: ${{ secrets.TEST_PYPI_API_TOKEN }}

- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    password: ${{ secrets.PYPI_API_TOKEN }}
```

**For this project:** Binaries are the distribution method ✅

---

### 10. Package Files and Binaries Summary

#### What Gets Built and Released

| File | Size | Type | Platform |
|------|------|------|----------|
| `butterfly-classifier-linux` | 125 MB | Executable | Linux (x64) |
| `butterfly-classifier-linux.sha256` | 65 bytes | Checksum | Verification |
| `butterfly-classifier-windows.exe` | 140 MB | Executable | Windows (x64) |
| `butterfly-classifier-windows.exe.sha256` | 73 bytes | Checksum | Verification |
| `butterfly-classifier-macos` | 118 MB | Executable | macOS (x64) |
| `butterfly-classifier-macos.sha256` | 68 bytes | Checksum | Verification |

#### How to Use the Release

**User downloads v0.1.0:**
```bash
# Download the executable for their OS
curl -L -O https://github.com/gurovamr/Butterfly-Image-Classification/releases/download/v0.1.0/butterfly-classifier-linux

# Make it executable
chmod +x butterfly-classifier-linux

# Run it (no Python installation needed!)
./butterfly-classifier-linux
```

---

## Part III: Additional CI Features

### 11. Artifact Management

#### Test Results Artifacts
```yaml
- name: Upload test results artifact
  uses: actions/upload-artifact@v4
  with:
    name: pytest-results
    path: junit/test-results.xml
```
- Stored for 90 days (default)
- Downloadable from Actions tab
- Useful for debugging CI failures

#### Binary Artifacts
```yaml
- name: Upload binary artifact
  uses: actions/upload-artifact@v4
  with:
    name: butterfly-classifier-${{ matrix.target }}
    path: release/
```
- Temporary storage before GitHub Release is created
- Allows manual download if release creation fails

---

### 12. Dependency Management (Bonus)

**File:** `.github/dependabot.yml`

Automatically creates PRs for dependency updates.

#### Configuration
```yaml
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
```

#### What Happens Weekly
1. Dependabot checks for new versions
2. Creates PRs for dependency updates
3. CI runs automatically on these PRs
4. If all checks pass, PRs are ready to merge
5. Updates are batched (max 10 open at once)

**Example Dependabot PR:**
```
Bump tensorflow from 2.21.0 to 2.22.0

Updates the requirements on tensorflow to permit the latest version.
Release notes: https://github.com/tensorflow/tensorflow/releases/tag/v2.22.0
Changelog: https://github.com/tensorflow/tensorflow/blob/v2.22.0/RELEASE.md

☑️ All checks passed
```

---

## Complete CI/CD Workflow Summary

### How Code Flows Through the System

```
1. Developer pushes code
   ↓
2. GitHub Actions: CI workflow starts
   ├─ Lint job (ruff + black checks)
   │  ├─ ✅ Pass → proceed to test
   │  └─ ❌ Fail → stop, require fixes
   │
   └─ Test job (pytest + coverage)
      ├─ Run 145 unit tests
      ├─ Run doctests in docstrings
      ├─ Measure coverage (must be ≥80%)
      ├─ Upload XML results
      ├─ Publish to PR
      └─ Upload coverage to Codecov
   
   ├─ Matrix job (Python 3.10/3.11/3.12)
   │  └─ Test on 3 versions in parallel
   │
   └─ Pylint job (code quality check)
      └─ Score must be ≥8.0

3. PR Status:
   ✅ All checks passed → Ready to merge
   ❌ Failed checks → Require fixes

4. Merge to main
   ↓
5. Developer creates release tag (v0.1.0)
   ↓
6. GitHub Actions: Release workflow starts
   ├─ Build binaries for Linux/Windows/macOS
   ├─ Generate SHA256 checksums
   ├─ Create GitHub Release
   └─ Upload all files
   
7. Release published → Users can download executables
```

---

## Key Metrics & Monitoring

### What Gets Tracked

| Metric | Tool | Target | Status |
|--------|------|--------|--------|
| Code Coverage | Codecov | ≥80% | ✅ Enforced |
| Lint Score | Ruff | 0 violations | ✅ Enforced |
| Format Compliance | Black | 100% | ✅ Enforced |
| Code Quality | Pylint | ≥8.0 | ✅ Enforced |
| Test Pass Rate | Pytest | 100% | ✅ Enforced |
| Python Compatibility | Matrix | 3.10, 3.11, 3.12 | ✅ Tested |

### Dashboard Views

- **GitHub Actions Tab:** Real-time workflow runs and logs
- **Codecov Dashboard:** Coverage trends over time
- **PR Checks:** Instant feedback on every PR
- **Release Page:** Binary downloads and release notes

---

## How to Trigger Workflows

### Manual Trigger (workflow_dispatch)

Some workflows can be triggered manually from GitHub:

1. Go to **Actions** tab
2. Select workflow (CI or Build Binaries)
3. Click **Run workflow**
4. Workflow starts immediately

### Create a Release (Tag Push)

```bash
# On your local machine
git tag v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

**Result:** Build binaries workflow triggers automatically

---

## Conclusion

This project implements a **comprehensive CI/CD pipeline** covering:

✅ **Continuous Integration:**
- Automated linting, formatting, testing
- Code coverage enforcement (80% minimum)
- Multi-version Python testing (3.10, 3.11, 3.12)
- Coverage reporting to Codecov
- Automatic test result publishing to PRs

✅ **Continuous Deployment:**
- Automated release workflow on version tags
- PyInstaller binary compilation for 3 platforms
- SHA256 checksum generation for integrity verification
- Automatic GitHub Release creation with assets
- Auto-generated release notes from commits

✅ **Dependency Management:**
- Weekly Dependabot updates
- Automatic PR creation for dependency upgrades

**Result:** High-quality, reliable software with zero manual release steps.
