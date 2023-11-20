class x:
    def __init__(self):
        self.hi = "hi"
    def __call__(self):
        print(self.hi)

test = x()
test.kk = "kk"
print(test.kk)