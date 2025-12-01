#  import base image
FROM python:3.11-slim

# create a work directory 
WORKDIR /app

# copy our file and folders into the work directory that we created
COPY . /app

# install the necessary dependencies
RUN pip install -r requirements.txt

# expose the port
EXPOSE 8000

# run your application
CMD ["uvicorn", "deploy.inference:app", "--host", "0.0.0.0", "--port", "8000"]