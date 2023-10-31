FROM ubuntu

# Update the package list
RUN apt-get update -y

# Install python & pip
RUN apt-get install -y python3 python3-pip

# Create directory
RUN mkdir ./home/doc-bd-a1

WORKDIR /home/doc-bd-a1

# Move the dataset & requirements into container
COPY requirements.txt .
COPY marketing_campaign.csv .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Open bash shell upon container startup
CMD ["/bin/bash"]