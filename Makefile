PYTHON=python
PIP=pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m py_compile src/train.py

test:
	@echo "Ejecutando pruebas básicas..."
	$(PYTHON) -c "import os; assert os.path.exists('config.yaml'), 'Falta config.yaml'"
	$(PYTHON) -c "import os; assert os.path.exists('data/ev_market_2026.csv'), 'Falta dataset'"
	$(PYTHON) -c "import os; assert os.path.exists('src/train.py'), 'Falta train.py'"
	@echo "Pruebas básicas finalizadas correctamente"

train:
	$(PYTHON) src/train.py

all: install lint test train