# code2doc

A toolkit for automatic code documentation generation. This repository takes un-annotated code from the `examples` directory, trains a custom model, and generates readable documentation for each code file.

## Table of Contents

- [code2doc](#code2doc)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Research Paper](#research-paper)
  - [Getting Started](#getting-started)
    - [Clone the repository](#clone-the-repository)
    - [Create Virtual Environment](#create-virtual-environment)
    - [Install Requirements](#install-requirements)
  - [Usage](#usage)
    - [1. Train the Model](#1-train-the-model)
    - [2. Generate Documentation](#2-generate-documentation)
  - [Project Structure](#project-structure)

## About

**code2doc** provides a pipeline to automatically generate documentation for Python code using a model trained on example scripts. The research paper included in this repository explains the motivation, methodology, and results.

## Research Paper

Please refer to [research_paper.pdf](research_paper.pdf) in this repository for a detailed explanation of the approach, experiments, and findings behind **code2doc**.

## Getting Started

### Clone the repository

```bash
git clone https://github.com/YARE0909/code2doc.git
cd code2doc
```

### Create Virtual Environment

It is recommended to use a virtual environment to avoid package conflicts:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Install Requirements

Make sure you have pip up to date. Then install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

Run `main.py` to train the code documentation model.

```bash
python main.py
```

### 2. Generate Documentation

After training, generate documentation for the code files within the `examples` directory by running:

```bash
python generate.py
```

The generated documentation will be saved in the `output` directory.

## Project Structure

```
code2doc/
├── examples/             # Input example codes (without docstrings)
├── images/               # Supporting images for this project
├── output/               # Generated documentation will appear here along with training outputs
├── eda.py                # Exploratory Data Analysis scripts
├── generate.py           # Script to generate documentation
├── main.py               # Script to train the model
├── requirements.txt      # Python dependencies
├── research_paper.pdf    # Research paper describing the project
├── .gitignore
└── ...
```

**For full details on methodology and experiments, see [research_paper.pdf](research_paper.pdf).**

[1] https://github.com/YARE0909/code2doc