.PHONY: build run clean test

build:
    docker build -t tiny_attention_networks:latest .

run:
    docker run --gpus all -it --rm \
        -p 6006:6006 \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/data:/app/data \
        tiny_attention_networks:latest

clean:
    docker system prune -f

test:
    docker run --gpus all -it --rm \
        -v $(pwd):/app \
        tiny_attention_networks:latest pytest
