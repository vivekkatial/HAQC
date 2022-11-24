# HAQC -- Heuristic Algorithms for Quantum Computing Research Group

![build](https://github.com/vivekkatial/HAQC/actions/workflows/build-container-and-release.yml/badge.svg)

Research group to run optimisation algorithms on Quantum Computers at the University of Melbourne 

## Getting Started

Before getting started, ensure you have Python 3.7+. We use [poetry](https://python-poetry.org/) to manage the python environment (the .gitignore file should already ignore it).

```{shell}
$ poetry install
```

To add a package to your new project:

```{shell}
$ poetry install <package>
```

This will automatically edit your `pyproject.toml` file with the new package you provided.

Next, activate the `poetry` shell:

```{shell}
$ poetry shell
$ python --version
```

This will spawn a new shell subprocess, which can be deactivated by using exit.

## Contributing

### Testing 

For testing, we use `pytest`. To run the tests, just type the command `pytest`, or you can specify a file e.g. `pytest tests/test_graph_generator.py`.

We will use `black` as our code formatter. Simply run `black -S .` to run black over all the files before committing. The `-S` is to skip string normalisation, because we prefer single quotes/don't really care ([flame war, I know](https://github.com/psf/black/issues/118)).

### Before making a PR

In summary, before merging a PR, you should:

```bash
# Make sure all tests pass
cd src
pipenv run python -m pytest tests/*

# Format with black
pipenv run python -m black -S .
```

## MLFlow Tracking

To get the MLFlow tracking functionality to work you will need to setup `awscli` credentials, so MLFlow can properly log artifacts.

If you're keen to do this then please follow the instructions [here](https://wiki-rcs.unimelb.edu.au/display/RCS/AWS+CLI)

You can request the credentials for this experiment from Vivek at vkatial@student.unimelb.edu.au

## Running a test instance

To run a test instance try out the steps below:

```bash
python qaoa_vrp/main.py -f test -T False # -T tracking for MLFlow
```

### Jupyter Notebooks

First ensure that your Python is _not_ aliased in your `.bashrc` or `.zshrc` file.

After this launch your `poetry` by

```{shell}
poetry shell
```

Then do:

```{shell}
python -m ipykernel install --user --name=ENV_NAME
```

Then launch the notebook

```{shell}
jupyter notebook
```

In your notebook, Kernel -> Change Kernel. Your kernel should now be an option.

<img src='images/jupyter-install.png'/>

## Singularity

This project leverages [Singularity](https://github.com/singularityhub/) to ensure the code is reproducible and manage dependencies. 

You can find the recipe for our container in `SingularityFile.def`. There are various apps for each different type of experiment we run,


## CI/CD 

We use Github Actions for CI/CD. Everytime a PR is created, a test build of the singularity container runs. When merging into `main` we do a release of the container.

## Authors
- Vivek Katial
