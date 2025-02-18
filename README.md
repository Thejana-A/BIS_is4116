# BIS_is4116

**Overview
**
- Business domain associated with this analytical process is sale of cars in USA.
- Business problem is identifying the factors that affected the prices of cars in USA, in 2024, to identify trends associated with prices of cars.
- This data set of cars contains following attributes:
    Brand (text): The brand or manufacturer of the car.
    	Ex: "Toyota", "Nissan"
    Model (text): The specific model of the car.
    	Ex: "Premio", "Allion"
    Mileage (numeric): The number of miles the car has been driven.
    	Ex: 102345, 247589
    Year (numeric): The manufacturing year of the car.
    	Ex: 2023, 2000
    Status (text): Indicates whether the car is new, used, or certified pre-owned.
    	Ex: "New", "Certified", "Used"
    Dealer (text): Information about the dealer or seller offering the car.
    	Ex: "Streamline Auto Outlet", "Ray Chevrolet"
    Price (numeric): The listed price of the car in USD.
 	    Ex: 25020, 33450
 	
- Since data associated with cars are highly diversified, it is difficult to fill missing data with substitute values. Therefore, data points associated with missing data were totally removed. (Removed percentage of data from initial data set = 0.026%)
- After pre-processing, 140956 data points were available for analysis.

**About the repository
**
Content of repository
- 20020015_is4116.ipynb
Jupyter Notebook file of project, which can be executed after opening with Jupyter Notebook application
- 20020015_is4116.pdf
Results obtained by author, by running "20020015_is4116.ipynb" file in Jupyter Notebook (This file is for users' reference)
- 20020015_is4116.py
Python file, containing code blockes present in cells of "20020015_is4116.ipynb" file
- car_sale_usa.csv
Pre-processed data set used for analytical process in "20020015_is4116.ipynb" file (in csv format)
- car_sale_usa.ods
Pre-processed data set used for analytical process in "20020015_is4116.ipynb" file (in spreadsheet format)

Note: 
- In PDF report submitted to VLE ("20020015.pdf", which is not present in this repository), there are 15 steps in the analytical process, identified with numbers from 1 to 15. 
- In "20020015_is4116.ipynb" file, in each cell excluding "#Loading data" cell, there is a comment as "#Analysis *" (* means any number from 1 to 15, where some of them have multiple parts as part_01, part_02 etc.). Those analysis numbers in "20020015_is4116.ipynb" correspond to respective steps in "20020015.pdf" PDF document.
Ex: 
- Number 01 in PDF document -> #Analysis 01 in "20020015_is4116.ipynb"
- Number 02 in PDF document -> #Analysis 03 in "20020015_is4116.ipynb"
- Number 04 in PDF document -> #Analysis 04_part_01 and #Analysis 04_part_02 in "20020015_is4116.ipynb"

**Steps to reproduce the project
**
Option 01:
1) Clone the repository to local machine
2) Navigate into "BIS_is4116" folder
3) Open "20020015_is4116.ipynb" file with Jupyter Notebook
4) Run each cell, starting from the one at top, until the last cell

Option 02:
1) Clone the repository to local machine
2) Navigate into "BIS_is4116" folder
3) Open Jupyter Notebook, and create a new Notebook.
4) Copy python code blocks from "20020015_is4116.py" file, starting from "#Loading data" until "#Analysis 15_part_02", such that each code block resides in a cell in Jupyter Notebook.
5) Run each cell, starting from "#Loading data", until "#Analysis 15_part_02"

Important:
- After cloning the repository to your local machine, make sure that you have required permissions to open files in it, and to execute them, as required.
- Make sure that "car_sale_usa.csv" file is available at the same level, as the Jupyter Notebook file in which you run these codes.

**Findings of the project
**
- New cars are more expensive than used cars.
- It can be expected to have a higher price for a car of any brand, if year of manufacture is later.
- Majority of cars are present in price range of $0-$100,000.
- It can be observed that with increase of mileage, price of car decreases.
- Even in same brand, mean price can vary from dealer to dealer. It means that, dealers tend to deal different car models, of same brand.
- In a certain brand, mean price of a car can vary significantly, from model to model.
