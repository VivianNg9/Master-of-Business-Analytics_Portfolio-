from mrjob.job import MRJob

class MRWordCount(MRJob):
    def mapper(self, _, line):
        yield line, 1

    def reducer(self, year_company, count):
        yield year_company, sum(count)

if __name__ == "__main__":
    MRWordCount.run()