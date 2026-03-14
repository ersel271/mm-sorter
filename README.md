# mm-sorter

#### 1. clone the repository

```bash
git clone <repo-url>
cd mm-sorter
```

#### 2. create a virtualenv (and activate it) 

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### 3. install dependencies (can be skipped on Linux / macOS since setup script installs them)

for dev environment (includes test tools)

```bash
pip install -r requirements-dev.txt
```

for prod environment (only runtime dependencies):

```bash
pip install -r requirements.txt
```

#### 4. run the setup script

```bash
./tools/setup.sh dev
```

#### 5. create a configuration from template

copy template configuration

Linux / macOS:

```bash
cp config/config.yaml.template config/config.yaml
```

Windows:

```cmd
copy config\config.yaml.template config\config.yaml
```

then modify camera and uart settings for your specific setup.

#### 6. run the tests

```bash
python -m pytest
```

to run with coverage and HTML reports:

```bash
./tools/run_tests.sh
```

reports are written to the `_report/` directory.