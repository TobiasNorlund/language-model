ARG TAG
FROM tensorflow/tensorflow:$TAG
RUN echo $TAG

RUN mkdir -p /opt/project/src
WORKDIR /opt/project/src

RUN apt update && apt install -y jq wget

COPY requirements.txt .
RUN pip install -r requirements.txt