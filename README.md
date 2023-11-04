# Data_Analysis_Pipeline

This is an exercise to practice the real-world data science operational flow.

# Files description
`Dockerfile` Specifying the base image as Ubuntu and installs Python3 with the required packages. It then creates a directory inside the container at /home/doc-bd-a1/ and copies the dataset into it and opens the bash shell upon startup.

`load.py` Dynamically reads the dataset file by accepting the file path as a user-provided argument.

`dpre.py` Performs example Data Cleaning, Data Transformation, Data Reduction, and Data Discretization steps then saves the resulting data frame as a new CSV file named res_dpre.csv. 

`eda.py` Conducts exploratory data analysis, generating at least 3 insights without visualizations then saves these insights as text files named eda-in-1.txt, and so on.

`vis.py` Creates a single visualization and save it as vis.png.

`model.py` Implements the K-means algorithm on the data frame then saves the number of records in each cluster as a text file named k.txt.

`final.sh` Is a simple bash script to copy the output files generated by dpre.py, eda.py, vis.py, and model.py from the container to the local machine in bd-a1/service-result/ then closes the container.

**Note:** Each one of these files automatically invokes the next one.

# Getting Started
Run the docker container then go into the analysis directory and run the following command
`python3 load.py <dataset_path>`

## Customer Personality Analysis Dataset Documentation
**Kaggle link:** https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
### Context
Customer Personality Analysis is a detailed analysis of a company's ideal customers. It helps a business better understand its customers and tailor products and marketing efforts to meet the specific needs, behaviors, and concerns of different types of customers.

### Problem Statement
The objective of this dataset is to analyze customer personalities and segment them based on their attributes and behavior. By doing so, businesses can make informed decisions on product modifications and targeted marketing campaigns.

## Content
### Attributes
#### People
- **ID**: Customer's unique identifier
- **Year_Birth**: Customer's birth year
- **Education**: Customer's education level
- **Marital_Status**: Customer's marital status
- **Income**: Customer's yearly household income
- **Kidhome**: Number of children in the customer's household
- **Teenhome**: Number of teenagers in the customer's household
- **Dt_Customer**: Date of customer's enrollment with the company
- **Recency**: Number of days since the customer's last purchase
- **Complain**: 1 if the customer complained in the last 2 years, 0 otherwise

#### Products
- **MntWines**: Amount spent on wine in the last 2 years
- **MntFruits**: Amount spent on fruits in the last 2 years
- **MntMeatProducts**: Amount spent on meat in the last 2 years
- **MntFishProducts**: Amount spent on fish in the last 2 years
- **MntSweetProducts**: Amount spent on sweets in the last 2 years
- **MntGoldProds**: Amount spent on gold in the last 2 years

#### Promotion
- **NumDealsPurchases**: Number of purchases made with a discount
- **AcceptedCmp1**: 1 if the customer accepted the offer in the 1st campaign, 0 otherwise
- **AcceptedCmp2**: 1 if the customer accepted the offer in the 2nd campaign, 0 otherwise
- **AcceptedCmp3**: 1 if the customer accepted the offer in the 3rd campaign, 0 otherwise
- **AcceptedCmp4**: 1 if the customer accepted the offer in the 4th campaign, 0 otherwise
- **AcceptedCmp5**: 1 if the customer accepted the offer in the 5th campaign, 0 otherwise
- **Response**: 1 if the customer accepted the offer in the last campaign, 0 otherwise

#### Place
- **NumWebPurchases**: Number of purchases made through the company's website
- **NumCatalogPurchases**: Number of purchases made using a catalog
- **NumStorePurchases**: Number of purchases made directly in stores
- **NumWebVisitsMonth**: Number of visits to the company's website in the last month

## Target
The objective is to perform clustering to summarize customer segments based on the provided attributes.

## Acknowledgement
The dataset for this project is provided by Dr. Omar Romero-Hernandez.
