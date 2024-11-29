# Makefile for setting up the environment, installing dependencies, and running scripts

# Variables
VENV_DIR = g15p3
DOWNLOAD_SCRIPT = src/Dataloader/download_data.py
DICOM2NIFTI_SCRIPT = src/Dataloader/custom.py
DATALOADER_SCRIPT = src/Dataloader/dataloader.py
EXP1_SCRIPT = src/exp_1/train_exp1.py
EXP1_1_SCRIPT = src/exp_1/train_exp1.1.py
EXP2_SCRIPT = src/exp_2/train_exp2.py
EXP2_1_SCRIPT = src/exp_2/train_exp2.1.py
REQUIREMENTS = requirements.txt

# Targets
.PHONY: all setup download dicom2nifti dataloader exp1 exp1.1 exp2 exp2.1 clean

all: setup download dicom2nifti dataloader exp1 exp1.1 exp2 exp2.1

setup:
	# Create virtual environment and install dependencies
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r $(REQUIREMENTS)

download:
	# Run the data download script
	$(VENV_DIR)/bin/python $(DOWNLOAD_SCRIPT)

dicom2nifti:
	# Convert DICOM files to NIfTI format
	$(VENV_DIR)/bin/python $(DICOM2NIFTI_SCRIPT)

dataloader:
	# Run the dataloader script
	$(VENV_DIR)/bin/python $(DATALOADER_SCRIPT)

exp1:
	# Run experiment 1 script
	$(VENV_DIR)/bin/python $(EXP1_SCRIPT)

exp1.1:
	# Run experiment 1.1 script
	$(VENV_DIR)/bin/python $(EXP1_1_SCRIPT)

exp2:
	# Run experiment 2 script
	$(VENV_DIR)/bin/python $(EXP2_SCRIPT)

exp2.1:
	# Run experiment 2.1 script
	$(VENV_DIR)/bin/python $(EXP2_1_SCRIPT)

clean:
	# Remove the virtual environment and temporary files
	rm -rf $(VENV_DIR)
	rm -rf ./NBIA-Download
