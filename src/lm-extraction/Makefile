.PHONY: all venv augment extract clean help

VIDEOSET  ?= videoset
VIDEO_OUT ?= video_out
NPY_OUT   ?= data 
VENV_DIR  ?= .venv

SUBDIRS := $(notdir $(wildcard $(VIDEOSET)/*/))

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Full pipeline: venv + augment + extract (default)"
	@echo "  venv     - Create Python venv and install dependencies"
	@echo "  augment  - Run video augmentation"
	@echo "  extract  - Run landmark extraction on augmented videos"
	@echo "  clean    - Remove venv, augmented videos, and npy output"
	@echo ""
	@echo "Variables:"
	@echo "  VIDEOSET=$(VIDEOSET)   INPUT videoset directory"
	@echo "  VIDEO_OUT=$(VIDEO_OUT) OUTPUT augmented videos directory"
	@echo "  NPY_OUT=$(NPY_OUT)     OUTPUT npy directory"
	@echo "  VENV_DIR=$(VENV_DIR)   Python venv directory"

all: venv augment extract

venv:
	python -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

augment: venv
	./video_augmentation.sh $(VIDEOSET)/ $(VIDEO_OUT)/

extract: augment
	$(foreach subdir,$(SUBDIRS),\
		$(VENV_DIR)/bin/python run_extraction.py batch $(VIDEO_OUT)/$(subdir)/ --out-npy-dir $(NPY_OUT)/$(subdir)/ &&) \
	true

clean:
	rm -rf $(VENV_DIR) $(VIDEO_OUT) $(NPY_OUT)
