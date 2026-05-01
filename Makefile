.PHONY: install app refresh test benchmark clean

install:
	python -m pip install -r requirements.txt

app:
	streamlit run dashboard/app.py

refresh:
	python run_pipeline.py

test:
	pytest -q

benchmark:
	python tools/benchmark.py --pipeline

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
