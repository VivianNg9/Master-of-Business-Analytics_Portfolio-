from mrjob.job import MRJob
from mrjob.step import MRStep

class MRBucketSort(MRJob):
 
    def configure_args(self):
        super(MRBucketSort, self).configure_args()
        self.add_passthru_arg('--num_buckets', type=int, default=250)
        self.add_passthru_arg('--bucket_size', type=int, default=3)
 
    def steps(self):
        return [
            MRStep(
                mapper=self.bucket_assignment_mapper
            ),
            MRStep(
                reducer=self.bucket_sort_reducer
            ),
            MRStep(
                reducer=self.bucketid_sort_reducer
            )
        ]

    def bucket_assignment_mapper(self, _, line):
        year_company, count = line.split('\t')
        count = int(count)
        bucket_id = count // self.options.bucket_size
        yield bucket_id, (year_company[1:-1], count)


    def bucket_sort_reducer(self, bucket_id, records):
        # Sort in descending order
        sorted_records = sorted(records, key=lambda x: (-x[1], x[0]))  
        for record in sorted_records:
            yield "key",(bucket_id, record)
    
    def bucketid_sort_reducer(self, key, bucketid_records):
        for value in sorted(bucketid_records, key=lambda x:x[0], reverse=True):
            yield value[1]

if __name__ == '__main__':
    MRBucketSort.run()