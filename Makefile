install:
	pyenv local rl_env
	pip install -r requirements.txt

pyenv:
	pyenv virtualenv 3.10.6 rl_env
