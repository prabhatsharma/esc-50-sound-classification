#!/bin/bash

aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 525158249545.dkr.ecr.us-west-2.amazonaws.com

docker build -t xeno-canto-melspec-creator .
docker tag xeno-canto-melspec-creator:latest 525158249545.dkr.ecr.us-west-2.amazonaws.com/xeno-canto-melspec-creator:latest
docker push 525158249545.dkr.ecr.us-west-2.amazonaws.com/xeno-canto-melspec-creator:latest



