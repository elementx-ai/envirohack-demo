FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

# expose
EXPOSE 7860

# set working directory
WORKDIR /app

# install pip and git
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y python3-pip wget ffmpeg

# install poetry and pytorch
RUN pip3 install --upgrade poetry

# add requirements
COPY ./pyproject.toml .
COPY ./poetry.lock .

# install requirements
RUN poetry run pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN poetry install --no-dev

# add source code
COPY . .

# install package
RUN poetry install --no-dev

# run server
CMD poetry run sed_demo