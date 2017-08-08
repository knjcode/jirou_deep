FROM knjcode/mxnet-finetuner:cpu

RUN cd /mxnet \
  && cd python \
  && pip3 install --upgrade pip \
  && pip3 install -e .

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8

COPY app.py .
COPY checker.py .

ENTRYPOINT [ "python3", "-u" ]
CMD [ "app.py" ]
