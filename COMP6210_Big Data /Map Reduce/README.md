# __<center>MapReduce Project</center>__

## __<center>Overview</center>__
This project focuses on the application of MapReduce paradigms for processing and analyzing a large movies dataset.

## __<center>Datasets</center>__
- `movies.zip`: Contains the movies dataset, which is used for all tasks in this project.

## __<center>Environment</center>__
- **MongoDB & Studio 3T:** For database creation and dataset importation.
- **pymongo:** To connect to MongoDB and extract document data.
- **mrjob:** Framework for implementing MapReduce programs.

## __<center>Details</center>__

### Task 1: Movie Count by Production Company and Year 
- **Objective:** Calculate the number of movies released by each production company for every year.
- **Steps:**
  1. **Data Retrieval:** Access 'movie -> date' to get release dates and extract the release year.
  2. **Company Identification:** Identify the top three production companies from 'movie -> companies'.
  3. **Data Formatting:** Create `<year, company>` pairs.
  4. **Data Storage:** Store these pairs in `year_and_company.txt`.
  5. **MapReduce Implementation:** Use mrjob to calculate the frequency of each `<year, company>` pair.

### Task 2: Sorting Algorithms Implementation 
- **Objective:** Implement Merge Sort and Bucket Sort algorithms using MapReduce to sort the results from Task 1.
  - **Merge Sort:** Sort `<year, company>` pairs in ascending order. Output stored in `task2_mergesort_output.txt`.
  - **Bucket Sort:** Sort `<year, company>` pairs in descending order. Output stored in `task2_bucketsort_output.txt`.

## Workflow
- **Task 1:**
  - `task1_extraction.py`: Extracts `<year, company>` pairs.
  - `task1_count.py`: MapReduce program to calculate the movie count by company and year.
- **Task 2:**
  - `task2_mergesort.py`: Implements Merge Sort.
  - `task2_bucketsort.py`: Implements Bucket Sort.

## Output
- [Report](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/Map%20Reduce/Map%20Reduce%20Report.pdf) with Flowchart and Pseudocode for each MapReduce program.
- Source code: `task1_extraction.py`, `task1_count.py`, `task2_mergesort.py`, `task2_bucketsort.py`.
- Output files: [`year_and_company.txt`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/Map%20Reduce/Output%20file%20for%20Task%201/year_and_company.txt), [`task1_output.txt`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/Map%20Reduce/Output%20file%20for%20Task%201/task1_output.txt), [`task2_mergesort_output.txt`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/Map%20Reduce/Output%20file%20for%20Task%202/task2_mergesort_output.txt), [`task2_bucketsort_output.txt`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/Map%20Reduce/Output%20file%20for%20Task%202/task2_bucketsort_output.txt).

