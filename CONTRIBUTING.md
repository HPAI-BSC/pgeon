# Contributing to pgeon / pgeon-xai

First off, thank you for considering contributing to pgeon! We welcome any contributions that help improve the project.

## Getting Started

To get your development environment set up, please follow these steps:

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:

    ```bash
    git clone https://github.com/HPAI-BSC/pgeon.git
    ```

3. **Change into the repository directory**:

    ```bash
    cd pgeon
    ```

4. **Create and sync a virtual environment**: We recommend using `uv`:

    ```bash
    uv sync --locked
    ```

5. **Install pre-commit hooks**: This helps ensure code style and quality consistency.

    ```bash
    uv run pre-commit install
    ```

## Contribution Workflow

1. **Create a new branch** for your feature or bug fix:

    ```bash
    git checkout -b your-branch-name
    ```

2. **Make your changes**: Write your code and add tests if applicable.
3. **Run tests**: Ensure all tests pass.

    ```bash
    uv run python -m unittest discover
    ```

4. **Commit your changes**: Use clear and descriptive commit messages. Pre-commit hooks will run automatically.

    ```bash
    git add .
    git commit -m "feat: Describe your feature or fix"
    ```

5. **Push your branch** to your fork:

    ```bash
    git push origin your-branch-name
    ```

6. **Open a Pull Request** (PR) against the `main` branch of the original `HPAI-BSC/pgeon` repository. Provide a clear description of your changes in the PR.

## Reporting Bugs or Requesting Features

Please use the [GitHub Issues](https://github.com/HPAI-BSC/pgeon/issues) page to report bugs or request new features. Provide as much detail as possible.

Thank you for contributing!
