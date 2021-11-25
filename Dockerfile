FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -f -y

RUN apt-get install wget python3 python3-pip python3-dev build-essential make cmake g++ libfreetype6-dev python3-matplotlib pkg-config -y

ADD ./requirements.txt /tmp/requirements.txt
RUN wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.1.0/en_core_web_lg-2.1.0.tar.gz
RUN pip3 install -r /tmp/requirements.txt

RUN mkdir /opt/BLINK
ADD ./ /opt/BLINK/

ENV PYTHONPATH=/opt/BLINK

EXPOSE 3030

WORKDIR /opt/BLINK

CMD ["python3", "/opt/BLINK/blink/main.py"]
