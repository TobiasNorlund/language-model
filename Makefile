.DEFAULT_GOAL := build

DATA_DIR:=/home/tobias/Data/transformer

GPU_TAG := nightly-gpu-py3
CPU_TAG := nightly-py3


# --- BUILD -----------------------

.PHONY: build-cpu
build-cpu:
	docker build --rm -t tobias/transformer-cpu --build-arg TAG=$(CPU_TAG) .

.PHONY: build-gpu
build-gpu:
	docker build --rm -t tobias/transformer-gpu --build-arg TAG=$(GPU_TAG) .

build: build-cpu build-gpu

# -- RUN --------------------------

.PHONY: run-cpu
run-cpu:
	docker run --rm -it -v $(CURDIR):/opt/project -v $(DATA_DIR):/data tobias/transformer-cpu bash

.PHONY: run-gpu
run-gpu:
	docker run --rm -it --runtime nvidia -v $(CURDIR):/opt/project -v $(DATA_DIR):/data tobias/transformer-gpu bash

