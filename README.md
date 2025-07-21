# skilang


## Project structure

Overview:
```bash
<root directory>
├── skilang/                # main package of the project
├── evaluation/             # evaluation package, for running benchmarks
├── models/                 # where the models trained models are stored
├── tests/                  # test package
├── .github/                # configuration of GitHub CI
│   └── workflows/          # configuration of GitHub Workflows
│       ├── check.yml       # runs tests on multiple OS and versions of Python
│       └── deploy.yml      # if check succeeds, and the current branch is one of {main, master}, triggers automatic releas on PyPi
├── LICENSE                 # license file
├── pyproject.toml          # project configuration file as prescribed by Poetry
├── renovate.json           # configuration of Renovate bot, for automatic dependency updates
├── requirements.txt        # only declares a dependency on Poetry.
└── release.config.js       # script to release on PyPi, and GitHub via semantic-release
```

## Requirements

The only requirement is Python 3.10 or later and Poetry, which needs to be installed with:
```
pip install -r requirements.txt
```

## Running the experiments

So far, it is possible to run the benchmarks for Adult use case, which uses the already trained models present in the [models](models) directory.

To run the benchmarks, you can use the following command:
```bash
poetry run python -m evaluation --spec-file ./tests/resources/adult.yml
```

You can find the results in the [evaluation/results](evaluation/results) directory.

Otherwise, if you want to exploit `skilang` to train a new model, you can use the following command:
```bash
poetry run python -m skilang --spec-file ./tests/resources/adult.yml
```

By default, models are saved in the [models](models) directory.
You can configure the output directory by modifying the destination field in the [adult.yml](tests/resources/adult.yml) specification file under the learnable section. 