import time

# Load the points from the dataset file
points = [] # Create a blank list to store data points
n=0
with open("R_tree_construction.txt", "r") as file:
    # Iterate over each line in the file
    for data in file.readlines():
        data = data.split()
        # Append a dictionary to 'points' with id, x, and y from the line, converting strings to integers
        points.append( 
            {
                "id": int(data[0]),  
                "x": int(data[1]),   
                "y": int(data[2])    
            }
        )


# Load the range queries
queries = [] # Create a blank list to store query ranges
with open("200Range.txt","r") as range:
    for r in range.readlines():
        r = r.split()  
        # Append a dictionary to 'queries' with x1, x2, y1, and y2 from the line, converting strings to integers
        queries.append({ 
            "x1": int(r[1]),  
            "x2": int(r[2]),  
            "y1": int(r[3]),  
            "y2": int(r[4])   
        })

# List to store the results of each query
results = []
start_time = time.time()  # Record the start time of the query processing

# Iterate through each query in 'queries'
for query in queries:
    count = 0  
    # Iterate through each point in 'points'
    for point in points:
        # Check if the point lies within the bounds of the query box
        if query["x1"] <= point["x"] <= query["x2"] and query["y1"] <= point["y"] <= query["y2"]:
            count += 1  # Increment counter if the point is within the query box
    results.append(count)  # Append the count of points in the query box to 'results'

end_time = time.time()  # Record the end time of the query processing
total_time = end_time - start_time  # Calculate total time taken for processing
average_time_per_query = total_time / len(results)  # Calculate average time per query

# Export the results to an output file
with open('Squential_output.txt', 'w') as f:
    f.write('Total time: ' + str(total_time) + ' seconds' + '\n') 
    f.write('Average time per query: ' + str(average_time_per_query) + ' seconds' + 2*'\n')
    f.write('Points count: ' + "\n")
    for count in results:
        f.write(str(count) + '\n')
