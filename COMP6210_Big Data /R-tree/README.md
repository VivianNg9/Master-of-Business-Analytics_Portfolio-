 # __<center>R-tree Implementation Project</center>__


## Overview
The project involves creating an R-tree for efficient spatial querying of 2D points. It emphasizes the correct application of R-trees in answering range queries and the comparative efficiency against sequential scanning.

## Dataset and Query Specification
- **Dataset:** A collection of 2D points, each with unique IDs and coordinates, provided in [`R_tree_Construction.txt`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/R-tree/R_tree_construction.txt).
- **Range Queries:** Set of 200 queries, each specifying a rectangular range, provided in [`200Range.txt`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/R-tree/200Range.txt).

## Environment and Languages
- **OS Environment:** MacOS
- **CPU:** Apple M2 Pro
- **RAM:** 16GB
- **Programming Language:** Python
- **Python Version:** 3.11.4
- **Development Tool:** Visual Studio Code

## Key Components of the Project
- **Sequential Scan:** Implemented in [`Sequential_Scan.py`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/R-tree/Sequential_Scan.py), this script processes spatial range queries sequentially and outputs the results.
- **R-Tree Construction and Querying:** Implemented in [`Rtree.py`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/R-tree/Rtree.py) and [`Rtree_DC.py`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/R-tree/Rtree_DC.py), these scripts construct R-trees and process queries utilizing a divide-and-conquer approach for improved efficiency.
- **Classes:** `Node` and `RTree`, essential for building and managing the R-tree structure.

## Output Files:
1. **Sequential Scan Output [`Sequential_output.txt`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/R-tree/Squential_output.txt):**
   - Contains the results of processing range queries using the sequential scan method.
   - Includes the total number of points within each query range.
   - Records the total and average query processing times.

2. **R-Tree Query Output [`Rtree_output.txt`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/R-tree/Rtree_output.txt):**
   - Results of the range queries processed using the R-tree.
   - Similar to the sequential output, it details the number of points per query range.
   - Provides insights into the efficiency of the R-tree method by listing total and average processing times.

3. **Divide-and-Conquer R-Tree Output [`Rtree_DC_output.txt`](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/R-tree/Rtree_DC_output.txt):**
   - Output from processing queries using the R-tree with a divide-and-conquer approach.
   - Highlights the performance improvement over the standard R-tree method.
   - Contains the count of points within each range query and the corresponding processing times.

4. **[Report](https://github.com/VivianNg9/Master-of-Business-Analytics_Portfolio-/blob/main/COMP6210_Big%20Data%20/R-tree/R_tree%20Report%20.pdf):**
   A comprehensive document detailing program functions, execution requirements, and a thorough analysis of R-Tree's construction and searching mechanism.





