# mm-sorter

#### 1. clone the repository

```bash
git clone <repo-url>
cd mm-sorter
```

#### 2. create a virtualenv (and activate it) 

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 4. run the setup script
setup script installs the requirements and creates some directories. it can adapt both dev and prod environments
```bash
./tools/setup.sh dev
```

#### 5. create a configuration from template

copy template configuration
```bash
cp config/config.yaml.template config/config.yaml
```

then modify any field for your specific setup

#### 6. run the tests

```bash
pytest
```

to run with coverage and HTML reports:

```bash
./tools/run_tests.sh
```

reports are written to the `_report/` directory

#### 7. finally, run the pipeline

if there are no problems in the test results, run the pipeline itself with
```bash
python3 sort.py
```

you can find out about options via
```bash
python3 sort.py --help
```
however, for a more detailed explanation about all the controls, reading the `sort.py` file docstring is recommended