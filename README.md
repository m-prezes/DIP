# DIP: Direct Interventions with Probing

Welcome to the **DIP** repository, dedicated to the experiments and research related to LLM interpretability methods for machine learning models. This repository is part of a Master Thesis project focusing on probing and directly intervening in models to better understand their decision-making processes.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Environment](#setting-up-the-environment)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

Before you begin, ensure you have met the following requirements:

- You are using a Linux-based system (Ubuntu recommended).
- You have `sudo` privileges on your machine.
- Python 3.10+ is installed on your system.

### Setting Up the Environment

1. **Install Make**

   `make` is a build automation tool that we use to streamline various setup and testing tasks. To install `make`, run the following command:

   ```bash
   sudo apt install make
   ```

2. **Install Poetry**

   `poetry` is a dependency management tool for Python projects. It simplifies the process of installing, managing, and packaging Python dependencies.

   To install `poetry`, run:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   export PATH="$HOME/.local/bin:$PATH"
   ```

Ensure that poetry is available in your terminal by running:

   ```bash
    poetry --version

   ```


## Usage

After setting up the environment, you can start running the experiments or tests defined in the repository. Detailed instructions and scripts for each experiment are available in their respective directories.

To activate the Python environment managed by `poetry`, use:

```bash
poetry shell
```

To install all required dependencies, run:
```bash
poetry install
```

Once the dependencies are installed, you can run the experiments or scripts provided in the repository. Typically, this involves navigating to the appropriate directory and executing the relevant script. Most experiments are organized as Jupyter notebooks.