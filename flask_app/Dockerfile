# start by pulling the python image
FROM python:3.8.8

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

RUN pip install -r requirements.txt

WORKDIR /keras-retinanet

RUN git clone https://github.com/fizyr/keras-retinanet.git .

#RUN git checkout 42068ef9e406602d92a1afe2ee7d470f7e9860df
#RUN python setup.py install

# Install retinaNet
RUN python setup.py build_ext --inplace
RUN pip install .

WORKDIR /app

#RUN export PATH=/user/local/bin:$PATH 

RUN apt-get update
RUN apt-get install tesseract-ocr -y 
RUN apt-get install tesseract-ocr-eng -y

RUN pip install pytesseract

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["app.py" ]