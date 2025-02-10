# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Gradio will run on
EXPOSE 7860

# Set Python to unbuffered mode
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "ui.py"]