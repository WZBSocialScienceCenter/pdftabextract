sdist:
	python setup.py sdist

wheel:
	python setup.py bdist_wheel

pypi_upload:
	python setup.py sdist bdist_wheel upload 

pypi_testregister:
	python setup.py register -r https://testpypi.python.org/pypi

pypi_testupload:
	python setup.py sdist bdist_wheel upload -r https://testpypi.python.org/pypi

