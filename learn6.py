class Fib( object ):
    def __init__(self  ):
        self.a,self.b=0,1
    def __iter__(self):
        return self
    def __next__(self):
        self.a,self.b=self.b,self.a+self.b
        if self.a>100000:
            raise StopIteration()
        return self.a
    def __getitem__(self, item):
        a,b=1,1
        for x in range( item ):
            a,b=b,a+b
        return a
class Student( object ):
    def __getattr__(self, item):
        if item=='age':
            return lambda : 22
        if item=='score':
            return 99
    def __call__(self  ):
        print("Call Test")
f=Fib()
s=Student()
print( f[5],f[6] )
print( s.age(),s.score )
s()
