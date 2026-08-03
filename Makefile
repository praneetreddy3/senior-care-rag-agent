.PHONY: install index test eval api ui mcp docker clean

install:
	pip install -r requirements.txt

index:
	python -m scripts.build_index

test:
	pytest -q

eval:
	python -m eval.run_eval --k 4 --json eval/results.json

api:
	uvicorn api:app --reload

ui:
	streamlit run app.py

mcp:
	python mcp_server.py

docker:
	docker build -t senior-care-rag-agent .

clean:
	rm -rf storage .pytest_cache __pycache__ src/__pycache__
