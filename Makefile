format:
	poetry run black .;
	poetry run isort .;

install:
	poetry install;