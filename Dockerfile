FROM python:3.11

WORKDIR /

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY ./app /

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]