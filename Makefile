.PHONY: run dev test build up

run:
\tuvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev:
\tENVIRONMENT=dev LOG_LEVEL=DEBUG make run

test:
\tpytest -q

build:
\tdocker build -t yolov8-api:latest .

up:
\tdocker compose up --build
