FROM python:3.7-slim

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -f -y

RUN apt-get install wget build-essential make cmake g++ -y

ADD ./requirements.txt /tmp/requirements.txt
RUN wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.1.0/en_core_web_lg-2.1.0.tar.gz
RUN pip3 install -r /tmp/requirements.txt

RUN mkdir /opt/BLINK
ADD ./ /opt/BLINK/

ENV PYTHONPATH=/opt/BLINK

EXPOSE 3030

WORKDIR /opt/BLINK

CMD ["python3", "/opt/BLINK/blink/main.py", "--fast", "--mode", "api"]
