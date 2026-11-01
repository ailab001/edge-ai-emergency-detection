.PHONY: up down test generate run-scenarios shell all clean

up:
	docker compose up -d --build edge

down:
	docker compose down

test: up
	docker compose exec edge python -m pytest edge_ai/tests/ -v

generate: up
	docker compose exec edge python -m edge_ai.datasets.generator

run-scenarios: generate
	docker compose exec edge python -m edge_ai.datasets.run_scenarios

shell: up
	docker compose exec edge bash

all: test run-scenarios

simulate: generate
	docker compose exec edge python -m edge_ai.datasets.simulate $(if $(SCENARIO),--scenario $(SCENARIO),) $(if $(CYCLE),--cycle $(CYCLE),)

clean: down
	rm -f edge_ai/datasets/*.json
