FROM python:3.8-alpine

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

COPY check_labels.py /check_labels.py

ENTRYPOINT ["/check_labels.py"]
