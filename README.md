# Radar Evaluator

[![CI](https://github.com/juliensimon/radar-evaluator/actions/workflows/ci.yml/badge.svg)](https://github.com/juliensimon/radar-evaluator/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](htmlcov/index.html)
[![Multi-Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-blue?logo=python&logoColor=white)](https://github.com/juliensimon/radar-evaluator/actions)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting](https://img.shields.io/badge/linting-flake8-blue.svg)](https://flake8.pycqa.org/)
[![Security](https://img.shields.io/badge/security-bandit-yellow.svg)](https://bandit.readthedocs.io/)
[![Type Check](https://img.shields.io/badge/type%20check-mypy-blue.svg)](https://mypy.readthedocs.io/)
[![Code Quality](https://img.shields.io/badge/code%20quality-A%2B-brightgreen.svg)](https://github.com/juliensimon/radar-evaluator)
[![Maintenance](https://img.shields.io/badge/maintained%3F-yes-green.svg)](https://github.com/juliensimon/radar-evaluator/graphs/commit-activity)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/juliensimon/radar-evaluator/pulls)
[![CLI](https://img.shields.io/badge/CLI-Command%20Line-black.svg)](https://en.wikipedia.org/wiki/Command-line_interface)
[![Linux](https://img.shields.io/badge/os-linux-blue?logo=linux)]()
[![macOS](https://img.shields.io/badge/os-macOS-lightgrey?logo=apple)]()
[![Windows](https://img.shields.io/badge/os-windows-blue?logo=windows)]()

---

A professional, extensible framework for evaluating LLMs across industries using radar charts and custom metrics.

## Features
- Multi-model, multi-industry evaluation
- Customizable metrics and questions
- Radar chart and summary report generation
- CLI for easy automation
- Extensible and well-tested

## Installation

```bash
# Clone the repository
$ git clone https://github.com/juliensimon/radar-evaluator.git
$ cd radar-evaluator

# (Optional) Create and activate a virtual environment
$ python3 -m venv .venv
$ source .venv/bin/activate

# Install dependencies using pyproject.toml
$ pip install .
```

## Usage

```bash
python radar_evaluator.py --model1 <model1> --model2 <model2> [--industries <industries>] [--num-questions <N>]
```

See `python radar_evaluator.py --help` for all options.

## Configuration
- Edit `config.json` to customize models, metrics, and evaluation parameters.
- Add industry-specific questions in `industry_questions.json`.

## Testing & Coverage

```bash
# Run tests
python -m pytest

# Run coverage
python -m pytest --cov=radar_evaluator

# View HTML coverage report
open htmlcov/index.html
```

## Contributing
- PRs are welcome! Please lint and test before submitting.
- Pre-commit hooks and style checks are enabled.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Acknowledgements
- [Black](https://github.com/psf/black)
- [Flake8](https://flake8.pycqa.org/)
- [Bandit](https://bandit.readthedocs.io/)
- [mypy](https://mypy.readthedocs.io/)
- [pytest](https://pytest.org/)
- [matplotlib](https://matplotlib.org/)
- [pandas](https://pandas.pydata.org/)
- [requests](https://requests.readthedocs.io/)
- [huggingface-hub](https://pypi.org/project/huggingface-hub/)
