# Use the official Python image as a base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the application code into the container
COPY higherEd.py .

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME=text_processing_app

# Run the application when the container launches
CMD ["python", "higherEd.py"]