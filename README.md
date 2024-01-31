henry
==============================

Gravitational-wave noise modelling with autoencoders

<!-- Machine learning algorithm to remove non-linear noise in gravitational-wave data around GW200129. Full reproduction requires access to LIGO internal data and a GPU with about 30GB memory. Production-quality model training takes about 30 minutes on Nvidia A100 80GB. -->

<!-- For more details about gravitational-wave noise, read for example [arXiv:2311.09921](https://arxiv.org/abs/2311.09921). --> 

![Image Alt text](/reports/figures/64_5_7.png)

Examples of the cleaned data are in the `reports/figures` folder.


Installation:
--------
1) Create a Conda environment with `make create_environment`
2) Install Python packages with `make requirements`
3) Install the `henry` package with `pip install .`

--------
Makefile commands:
- `create_environment` - Set up Python interpreter environment.
- `requirements` - Set up Python dependencies.
- `test_environment` - Test Python environment setup.
- `train` - Train the model.
- `test` - Clean the test data set containing glitches.
- `clean` - Delete all compiled Python files.
- `lint` - Lint using flake8.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

