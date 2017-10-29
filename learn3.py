import functools
def log( *args ):
    def decorator( func ):
        @functools.wraps(func)
        def wrapper():
            print('%s %s():' %( args[0] if len( args )>0 else "call" ,func.__name__ ))
            func()
            print('%s %s():' % (args[0] if len(args) > 0 else "call", func.__name__))
        return wrapper
    return decorator
@log()
def now():
    print( "ssss" )
#print( now() )
int=functools.partial( int,base=2 )
print( int( '1010111') )
print( int( '1010111',base=10) )
kw={'base':2}
print( int( '10010',**kw ) )
max2=functools.partial( max,10 )
print( max2( 5,6 ) )
