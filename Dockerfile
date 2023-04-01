FROM tensorflow/tensorflow:2.11.0-gpu
# FROM tensorflow/tensorflow:2.9.1-gpu <-- THIS WORKS FOR TRAINING
# ^ For GPU tf

# FROM tensorflow/tensorflow:2.9.1
# ^ For CPU tf

RUN apt-get update -qq \
    && apt-get install -yq python3-venv a2ps parallel\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    && apt-get install ffmpeg libsm6 libxext6  -y

RUN python -m venv --system-site-packages /env
ENV VIRTUAL_ENV /env
ENV PATH "/env/bin:$PATH"

ADD requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


WORKDIR /app

ENTRYPOINT ["bash", "-c"]
