#syntax=docker/dockerfile:experimental
FROM nvcr.io/nvidia/deepstream-l4t:6.1-base

ENV GI_TYPELIB_PATH /usr/lib/aarch64-linux-gnu/girepository-1.0/

RUN apt-get update && \