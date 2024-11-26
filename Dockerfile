FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment.yml file
COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml

# Upgrade pip inside the new environment
RUN conda run -n library-analyzer pip install --upgrade pip

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "library-analyzer", "/bin/bash", "-c"]

# Copy the application code
COPY . .

# Install any remaining dependencies
RUN pip install -r requirements.txt

# Set the entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "library-analyzer", "python", "-m", "library_analyzer"]

# Allow arguments to be passed to the entrypoint
CMD []

