from dataset import median

# helper function
def make_binary(target):
    return True if target > median else False

# data pipeline
def doItForTheWine(wines):

    # convert quality into labels (True if quality is higher than median of dataset, else False)
    wines = wines.map(lambda input, target: (input, make_binary(target)))

    # cache progress in memory
    wines = wines.cache()

    # shuffle, batch and prefatch
    wines = wines.shuffle(1000)
    wines = wines.batch(32)
    wines = wines.prefetch(20)

    # return preprocessed data
    return wines