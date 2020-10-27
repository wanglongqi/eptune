.ONESHELL:
SHELL := /bin/bash
SRC = $(shell find nbs \( ! -regex '.*/\..*' \) -name "*.ipynb")

all: ept docs

ept: $(SRC)
	nbdev_build_lib
	touch eptune

sync:
	nbdev_update_lib

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	find . -name ".ipynb_checkpoints" -type d -exec rm -r {} +
	nbdev_build_docs
	touch docs

test:
	nbdev_test_nbs

release: pypi
	nbdev_conda_package
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist
