help:
		@echo make env
		@echo make install
		@echo make data
		@echo make clear
		@echo make run
		@echo make run-ui
		@echo make clean

env:
		python3 -m venv --system-site-packages venv
		# . venv/bin/activate
		# run setenv.sh manually via . venv/bin/activate

install:
		python3 -m pip install --upgrade pip
		pip install -r requirements.txt

data:
		python3 data.py

clear:
		rm -fr VCF
run:
		python3 main.py

clean:
		rm -fr venv