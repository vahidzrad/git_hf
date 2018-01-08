#!/bin/bash
docker run --rm -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/cmaurini/gradient-damage:latest ". sourceme.sh; /bin/bash -i"
