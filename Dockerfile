# Python image to use.
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY ./requirements.txt /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common
#RUN add-apt-repository universe
RUN apt-get update && apt-get install -y \
    curl \
    git \  
    python3.4 \
    python3-pip


RUN apt-get install -y libportaudio2
RUN pip install -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Run app.py when the container launches
ENTRYPOINT ["python", "app.py"]
