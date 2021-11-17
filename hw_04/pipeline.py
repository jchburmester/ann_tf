from functions import make_binary

# data pipeline
def doItForTheWine(wines):

    # convert quality into labels (True if quality is higher than median of dataset, else False)
    wines = wines.map(lambda target: make_binary(target))

    # shuffle, batch and prefatch
    wines = wines.shuffle(1000)
    wines = wines.batch(32)
    wines = wines.prefetch(20)

    # return preprocessed data
    return wines