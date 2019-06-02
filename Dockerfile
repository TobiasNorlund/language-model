ARG TAG
FROM tensorflow/tensorflow:$TAG
RUN echo $TAG

WORKDIR /root

RUN apt update && apt install -y jq wget

COPY requirements.txt .
RUN pip install -r requirements.txt