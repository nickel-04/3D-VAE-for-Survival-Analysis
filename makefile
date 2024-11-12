# Makefile for setting up the environment, installing dependencies, and running the Python download script


VENV_DIR = g15p3
DOWNLOAD_SCRIPT = Dataset/custom_dataset.py
REQUIREMENTS = requirements.txt


.PHONY: all setup run clean


all: setup run


setup:
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS)


run:
	$(VENV_DIR)/bin/python $(DOWNLOAD_SCRIPT)


clean:
	rm -rf $(VENV_DIR)
	rm -rf ./NBIA-Download


