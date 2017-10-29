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
print( now() )
