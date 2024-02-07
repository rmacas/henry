henry
==============================

Gravitational-wave noise modeling with autoencoders. This particular model removes low-frequency blips. The model can be adapted with a few minor changes to remove other short-duration noise transients in gravitational-wave data, e.g., Tomte glitches.

For more details about gravitational-wave noise, read for example [arXiv:1602.03844](https://arxiv.org/abs/1602.03844) and [arXiv:1611.04596](https://arxiv.org/abs/1611.04596).

The model training takes less than 5 minutes on Nvidia A100 and requires less than 8GB of memory.

![Image Alt text](/reports/figures/64_5_7.png)

Examples of the cleaned data are in the `reports/figures` folder.

Installation
--------
1) Create a Conda environment with `make create_environment`
2) Install Python packages with `make requirements`
3) Install the `henry` package with `pip install .`

Makefile commands
--------
- `create_environment` - Set up Python interpreter environment.
- `requirements` - Set up Python dependencies.
- `test_environment` - Test Python environment setup.
- `train` - Train the model.
- `test` - Clean the test data set containing glitches.
- `clean` - Delete all compiled Python files.
- `lint` - Lint using flake8.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
