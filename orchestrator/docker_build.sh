#!/bin/sh
docker buildx build --platform linux/amd64,linux/arm64 -t minuk0815/micro-magentic-one-$agent:0.0.25 -f ./$agent/Dockerfile --push .