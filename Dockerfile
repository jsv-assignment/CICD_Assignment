FROM python:3.11

WORKDIR /jsvdocker

COPY . /jsvdocker

RUN pip install --no-cache-dir -r requirements.txt

RUN python train.py

CMD ["python", "test.py"]
