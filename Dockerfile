FROM tensorflow/tensorflow:2.4.0-gpu

MAINTAINER Illia Herasymenko, illia.cgerasimenko@gmail.com

RUN apt-get -y update && apt-get -y install xauth && add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get -y install python3.8 && apt -y upgrade

RUN mkdir -p /workdir/

WORKDIR /workdir/

COPY docker_requirements.txt .

RUN pip install --upgrade pip && pip install -r docker_requirements.txt

COPY . .

WORKDIR /workdir/src

CMD ["python", "./run.py"]