from pymongo import MongoClient 

# Making database connection
client = MongoClient('localhost', 27017)
db = client["assignment1"]
col = db["movies"]
cur = col.find() 

# Write a txt file with <year, company> pairs of all movies 
file = open("year_and_company.txt", "w")
# Loop through each movie document in the cur
for movie in cur: 
    for i in range(min(len(movie['companies']), 3)):
        year_company = movie['date'][-4:] + " , " + movie['companies'][i]['name']+ '\n'
    file.write(year_company)  
file.close()