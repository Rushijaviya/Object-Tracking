## Download and setup ObjectTracking

## Virtual Environments (optional)

We highly recommend installing `ObjectTracking` and its dependencies in a
[`venv`](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) Python virtual
environment. Just copy-paste the following snippet to create a virtual
environment:

**Virtual Environments For Windows**

```bash
#Installing virtualenv
py -m pip install --user virtualenv

# Create the virtual environment
py -m venv env

# Activating a virtual environment
.\env\Scripts\activate

# Activate the virtual environment
# This will add the prefix "(in-toto-demo)" to your shell prompt
source in-toto-demo/bin/activate
```

**Virtual Environments For Unix**

```bash
#Installing virtualenv
python3 -m pip install --user virtualenv

# Create the virtual environment
python3 -m venv env

# Activating a virtual environment
source env/bin/activate

# Activate the virtual environment
# This will add the prefix "(in-toto-demo)" to your shell prompt
source in-toto-demo/bin/activate
```

**Get demo files and install in ObjectTracking**

```bash
# Fetch the demo repo using git
git clone https://github.com/Rushijaviya/Object-Tracking.git

# Change into the demo directory
cd ObjectTracking

# Install a compatible version of in-toto
pip install -r requirements.txt
```

## Run the ObjectTracking commands

```shell
python manage.py runserver
```
