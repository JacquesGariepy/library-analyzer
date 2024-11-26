FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yml file
COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "library-analyzer", "/bin/bash", "-c"]

# Copy the application code
COPY . .

# Install any remaining dependencies
RUN pip install -r requirements.txt

# Set the entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "library-analyzer", "python", "use_case.py"]