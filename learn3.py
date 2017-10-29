def now():
    print( 'sjdkal' )
f=now
f()
print(f.__name__)
def log( func ):
    def wrapper( *args,**kw ):

