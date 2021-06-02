class AccessCounter:
    def __init__(
        self,
        args,
    ) -> None:
        self.args = args
        self.counts = dict()
        for key in range(args.num_parameters):
            self.counts[key] = 0


    def count(self, keys):
        for key in set(keys.flatten().numpy()):
            self.counts[int(key)] += 1
    
    
    def save(self, filename):
        print("saving to: " + filename)
        with open(filename, 'w') as out:
            for key in range(self.args.num_parameters):
                if key % 100000 == 0:
                    print(key)
                out.write(str(self.counts[key]) + "\n")
        print("saving done!")