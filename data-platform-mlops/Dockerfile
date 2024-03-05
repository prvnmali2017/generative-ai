FROM ghcr.io/mlflow/mlflow


WORKDIR /work

COPY . .


RUN apt-get update && \
    apt-get install -y curl unzip default-jdk && \
    rm '/usr/local/bin/databricks'

# Get databricks cli v0.208.1
RUN curl https://raw.githubusercontent.com/databricks/setup-cli/v0.208.1/install.sh | sh
