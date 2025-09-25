#!/bin/sh
docker buildx build --platform linux/amd64 -t minuk0815/micro-magentic-one-$agent --push .