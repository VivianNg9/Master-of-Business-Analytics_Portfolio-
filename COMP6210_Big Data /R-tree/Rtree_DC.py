import sys
import math
import time

B = 4
def main():
    points = [] # Create a blank list to store data points
    with open("R_tree_construction.txt", "r") as file:
        for data in file.readlines():
            data = data.split()
            # Append a dictionary to 'points' with id, x, and y from the line, converting strings to integers
            points.append({
                "id": int(data[0]),
                "x": int(data[1]),
                "y": int(data[2])
            }
        )
            print("x=", data[1], "y=", data[2])

# Load the range queries
    queries = [] # Create a blank list to store query ranges
    with open("200Range.txt", "r") as range:
        for r in range.readlines():
            r = r.split()
            # Append a dictionary to 'queries' with x1, x2, y1, and y2 from the line, converting strings to integers
            queries.append({
                "x1": int(r[1]),
                "x2": int(r[2]),
                "y1": int(r[3]),
                "y2": int(r[4])
        })
            print ("The current query is", queries[-1])
    
    points.sort(key=lambda point: point['x'])  # Sort data points based on x-value

    # Divide the sorted data points into two sets
    mid_index = len(points) // 2
    points1 = points[:mid_index]
    points2 = points[mid_index:]
    

    # Build R-Tree for first half
    rtree1 = RTree()
    # build R-Tree for second half
    rtree2 = RTree()


    print("build R-Tree: ")
    print("\n")

    R_Tree_start=time.time() # Record the start time of the query processing
    
    for point in points1: #insert data points from the root one by one 
        rtree1.insert(rtree1.root, point) 
    for point in points2: #insert data points from the root one by one 
        rtree2.insert(rtree2.root, point) 

    R_Tree_end = time.time()  # Record the end time of the query processing
    Used_time=R_Tree_end-R_Tree_start # Calculate total time to build rtree
    print("R-Tree construction completed\n")
    print("The time-cost of building up the R-Tree is",Used_time,"seconds.\n")

    results = [] # create a blank list to store the results
    Answer_query_start=time.time() # record start time for query
    for query in queries:
        # Query both R-Trees
        results_rtree1 = rtree1.query(rtree1.root, query)
        results_rtree2 = rtree2.query(rtree2.root, query)
        # Combine results (this step may vary depending on your specific requirements)
        combined_results = results_rtree1 + results_rtree2
        results.append(combined_results)

    Answer_query_end=time.time()
    Query_processing_time=Answer_query_end-Answer_query_start # calculate total query time
    print ("There are",results,"data points included in the query.\n")
    print ("The query processing time is",Query_processing_time,"seconds.")

# Create R_tree_output:
    with open('Rtree_DC_output.txt', 'w') as f: 
        f.write("The time-cost of building up the R-Tree is: " + str(Used_time) +" seconds.\n") # Print total time to build rtree
        f.write("The query processing time is: " + str(Query_processing_time) + " seconds.\n") # print total query time
        f.write("The average query processing time per point is: " + str(Query_processing_time/len(results)) + " seconds.\n") # Print average query time
        
        f.write('Points count: ' + "\n")
        for count in results:
            f.write(str(count) + '\n')


class Node(object): #node class
    def __init__(self):
        self.id = 0
        self.child_nodes = [] # list of child nodes ( for internal nodes)
        self.data_points = [] # list of datapoints (for leaf nodes)
        self.parent = None # parent node 
        self.MBR = { #MBR of the node (add value -1 for update later when build the rtree)
            'x1': -1,
            'y1': -1,
            'x2': -1,
            'y2': -1,
        }
    def perimeter(self): # calculate half permeter of the node
        return (self.MBR['x2'] - self.MBR['x1']) + (self.MBR['y2'] - self.MBR['y1'])

    def is_overflow(self): #Check overflow
        if self.is_leaf():
            if self.data_points.__len__() > B: # Check if number of datapoints > B
                return True
            else:
                return False
        else:
            if self.child_nodes.__len__() > B: #Check if number of child nodes > B
                return True
            else:
                return False

    def is_root(self): #Check if the node is root node
        if self.parent is None: #Is a root node if there is no parent node. Otherwise not root node
            return True
        else:
            return False

    def is_leaf(self): #Check if the node is leaf node
        if self.child_nodes.__len__() == 0: #Is leaf node if there is child node. Otherwise leaf node
            return True
        else:
            return False

class RTree(object): #R tree class
    def __init__(self):
        self.root = Node() #Create a root

    def query(self, node, query): #run to answer the query
        num = 0
        if node.is_leaf(): #If the node is leaf node, check the datapoints are in the node
            for point in node.data_points: #check if point in node covered by query
                if self.is_covered(point, query):
                    num = num + 1
            return num
        else:
            for child in node.child_nodes: #If it is not a leaf node, use MBR to check all the child nodes to see whether there is an intersection
                if self.is_intersect(child, query): #If there is an intesection, keep continue to check the child nodes in the next layer till the leaf nodes
                    num = num + self.query(child, query)
            return num

    def is_covered(self, point, query): #Check if points in nodes covered by the query range
        x1, x2, y1, y2 = query['x1'], query['x2'], query['y1'], query['y2']
        if x1 <= point['x'] <= x2 and y1 <= point['y'] <= y2: #Is covered if the x,y of points inside the x1-x2, y1-y2 range
            return True
        else:
            return False    

    def is_intersect(self, node, query): 
        center1_x = (node.MBR['x2'] + node.MBR['x1']) / 2 # Get the x of centerpoint of the node's MBR
        center1_y = (node.MBR['y2'] + node.MBR['y1']) / 2 # Get the y of centerpoint of the node's MBR
        length1 = node.MBR['x2'] - node.MBR['x1'] # Get the length of the node's MBR based on x axis
        width1 = node.MBR['y2'] - node.MBR['y1'] # Get the length of the node's MBR based on y axis
        center2_x = (query['x2'] + query['x1']) / 2 # Get the x of centerpoint of the query's MBR
        center2_y = (query['y2'] + query['y1']) / 2# Get the y of centerpoint of the query's MBR
        length2 = query['x2'] - query['x1'] # Get the length of the range's MBR based on x axis
        width2 = query['y2'] - query['y1'] # Get the length of the range's MBR based on y axis
        if abs(center1_x - center2_x) <= length1 / 2 + length2 / 2 and\
                abs(center1_y - center2_y) <= width1 / 2 + width2 / 2:  #Check if 2 MBR intersects ( distance between 2 centers smaller than length of MBR's sides)
            return True
        else:
            return False                    


    def insert(self, u, p): # insert p(data point) to u (MBR)
        if u.is_leaf(): #check if u is a leaf node
            self.add_data_point(u, p) #add the data point and update the corresponding MBR
            if u.is_overflow(): #Check overflow
                self.handle_overflow(u) #handle overflow for leaf nodes
        else:
            v = self.choose_subtree(u, p) #choose a subtree to insert the data point to miminize the perimeter sum
            self.insert(v, p) #keep continue to check the next layer recursively until it is leaf node
            self.update_mbr(v) #update the MBR for inserting the data point

    def choose_subtree(self, u, p): 
        if u.is_leaf(): #find the leaf and insert the data point
            return u
        else:
            min_increase = sys.maxsize #set an initial value for perimeter increasement
            best_child = None #define the best child node to insert point
            for child in u.child_nodes: #check each child to find the best node to insert the point 
                if min_increase > self.peri_increase(child, p): # check if the perimeter increasement is lower than the min_increase
                    min_increase = self.peri_increase(child, p)
                    best_child = child #change best_child to the child node which the peri_increase is the lowest
            return best_child

    def peri_increase(self, node, p): # calculate the increase of the perimeter after inserting the new data point
        # new perimeter(max/min of node's mbr and the new points) - original perimeter = increase of perimeter
        origin_mbr = node.MBR
        x1, x2, y1, y2 = origin_mbr['x1'], origin_mbr['x2'], origin_mbr['y1'], origin_mbr['y2']
        increase = (max([x1, x2, p['x']]) - min([x1, x2, p['x']]) +
                    max([y1, y2, p['y']]) - min([y1, y2, p['y']])) - node.perimeter()
        return increase


    def handle_overflow(self, u):
        u1, u2 = self.split(u) #u1 u2 are the two splits returned by the function "split"
        # if u is root, create a new root with s1 and s2 as its' children
        if u.is_root():
            new_root = Node() # create new node for the root
            self.add_child(new_root, u1) #add u1 as children to new root
            self.add_child(new_root, u2) #add u2 as children to new root
            self.root = new_root #update the root of the rtree 
            self.update_mbr(new_root)# update the mbr of the root node
        # if u is not root, delete u, and set s1 and s2 as u's parent's new children
        else:
            w = u.parent
            # copy the information of s1 into u
            w.child_nodes.remove(u)
            self.add_child(w, u1) #link the two splits and update the corresponding MBR
            self.add_child(w, u2)
            if w.is_overflow(): #check the parent node recursively for overflow
                self.handle_overflow(w)
            
    def split(self, u):
        # split u into s1 and s2
        best_s1 = Node() # create two nodes s1 and s2
        best_s2 = Node()
        best_perimeter = sys.maxsize
        # u is a leaf node
        if u.is_leaf():
            m = u.data_points.__len__()
            # create two different kinds of divides
            divides = [sorted(u.data_points, key=lambda data_point: data_point['x']),
                       sorted(u.data_points, key=lambda data_point: data_point['y'])] #sorting the points based on X dimension and Y dimension
            for divide in divides:
                for i in range(math.ceil(0.4 * B), m - math.ceil(0.4 * B) + 1): #check the combinations to find a near-optimal one
                    s1 = Node() #Create 1 node contains the first part 
                    s1.data_points = divide[0: i]
                    self.update_mbr(s1)
                    s2 = Node() #Create 1 node contains the second part
                    s2.data_points = divide[i: divide.__len__()]
                    self.update_mbr(s2)
                    if best_perimeter > s1.perimeter() + s2.perimeter(): #Find the best combination to minimize the total perimeter
                        best_perimeter = s1.perimeter() + s2.perimeter()
                        best_s1 = s1
                        best_s2 = s2

        # u is an internal node
        else:
            # create four different kinds of divides
            m = u.child_nodes.__len__()
            divides = [sorted(u.child_nodes, key=lambda child_node: child_node.MBR['x1']), #sorting based on MBRs
                       sorted(u.child_nodes, key=lambda child_node: child_node.MBR['x2']), 
                       sorted(u.child_nodes, key=lambda child_node: child_node.MBR['y1']),
                       sorted(u.child_nodes, key=lambda child_node: child_node.MBR['y2'])]
            for divide in divides:
                for i in range(math.ceil(0.4 * B), m - math.ceil(0.4 * B) + 1): #check the combinations
                    s1 = Node() #Create 1 node contains the first part 
                    s1.child_nodes = divide[0: i]
                    self.update_mbr(s1)
                    s2 = Node() #Create 1 node contains the second part
                    s2.child_nodes = divide[i: divide.__len__()]
                    self.update_mbr(s2)
                    if best_perimeter > s1.perimeter() + s2.perimeter(): #Find the best combination to minimize the total perimeter
                        best_perimeter = s1.perimeter() + s2.perimeter()
                        best_s1 = s1
                        best_s2 = s2

        for child in best_s1.child_nodes: # Update the parent node of the splitted node's children node
            child.parent = best_s1
        for child in best_s2.child_nodes: # Update the parent node of the splitted node's children node
            child.parent = best_s2

        return best_s1, best_s2


    def add_child(self, node, child):
        node.child_nodes.append(child) #add child nodes to the current parent (node) and update the MBRs. It is used in handeling overflows
        child.parent = node
        if child.MBR['x1'] < node.MBR['x1']: #Recalculate MBR
            node.MBR['x1'] = child.MBR['x1']
        if child.MBR['x2'] > node.MBR['x2']:
            node.MBR['x2'] = child.MBR['x2']
        if child.MBR['y1'] < node.MBR['y1']:
            node.MBR['y1'] = child.MBR['y1']
        if child.MBR['y2'] > node.MBR['y2']:
            node.MBR['y2'] = child.MBR['y2']
    # return the child whose MBR requires the minimum increase in perimeter to cover p

    def add_data_point(self, node, data_point): #add data points and update the the MBRS
        node.data_points.append(data_point)
        if data_point['x'] < node.MBR['x1']:
            node.MBR['x1'] = data_point['x']
        if data_point['x'] > node.MBR['x2']:
            node.MBR['x2'] = data_point['x']
        if data_point['y'] < node.MBR['y1']:
            node.MBR['y1'] = data_point['y']
        if data_point['y'] > node.MBR['y2']:
            node.MBR['y2'] = data_point['y']


    def update_mbr(self, node): #update MBRs when forming a new MBR. It is used in checking the combinations and update the root
        x_list = []
        y_list = []
        if node.is_leaf(): #Add x and y to x_list and y_list for leaf node
            x_list = [point['x'] for point in node.data_points]
            y_list = [point['y'] for point in node.data_points]
        else: #If node is not leaf node, add x and y of child node's MBR
            x_list = [child.MBR['x1'] for child in node.child_nodes] + [child.MBR['x2'] for child in node.child_nodes]
            y_list = [child.MBR['y1'] for child in node.child_nodes] + [child.MBR['y2'] for child in node.child_nodes]
        new_mbr = { #update MBR 
            'x1': min(x_list),
            'x2': max(x_list),
            'y1': min(y_list),
            'y2': max(y_list)
        }
        node.MBR = new_mbr

if __name__ == '__main__':
    main()