FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common curl git vim make
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install -y python3.10 python3.10-venv
RUN python3.10 -m ensurepip
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN ln -s /usr/local/bin/pip3.10 /usr/bin/pip
RUN mkdir /app
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN pip cache purge

CMD ./boot.sh
