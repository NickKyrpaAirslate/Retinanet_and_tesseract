# start by pulling the python image
FROM python:3.8.8
# copy the requirements file into the image (avoids being invalidated by the cache for every build)
COPY ./requirements.txt /app/requirements.txt

# switch working directory
WORKDIR /app

RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

EXPOSE 8000

CMD ["main.py" ]