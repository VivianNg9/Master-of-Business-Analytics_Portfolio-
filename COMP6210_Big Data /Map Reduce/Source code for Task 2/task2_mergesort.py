from mrjob.job import MRJob
from mrjob.step import MRStep

class MRMergeSort(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.mapper, reducer=self.reducer)
        ]

    def mapper(self, _, line):
        year_company, count = line.split('\t')
        count = int(count)
         # "key" for combining year_company and count 
        yield "key", (year_company[1:-1], count) 

    def merge_sort(self, arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            left_half = arr[:mid]
            right_half = arr[mid:]

            self.merge_sort(left_half)
            self.merge_sort(right_half)

            i = j = k = 0
            while i < len(left_half) and j < len(right_half):
                # Sort in ascending order
                if left_half[i][1] < right_half[j][1]:  
                    arr[k] = left_half[i]
                    i += 1
                else:
                    arr[k] = right_half[j]
                    j += 1
                k += 1

            while i < len(left_half):
                arr[k] = left_half[i]
                i += 1
                k += 1

            while j < len(right_half):
                arr[k] = right_half[j]
                j += 1
                k += 1

        # Return the sorted array
        return arr  

    def reducer(self, _, tuples):
        sorted_tuples = self.merge_sort(list(tuples))
        for tuple in sorted_tuples:
            yield tuple[0], tuple[1]

if __name__ == '__main__':
    MRMergeSort.run()