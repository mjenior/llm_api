VERSION = 1.3.10

bump:
	bumpversion --current-version ${VERSION} patch --allow-dirty --no-tag --no-commit

build:
	python3 -m build

clean:
	rm -rf promptpal.egg*

deploy: build clean
	python3 -m twine upload --repository pypi dist/*

