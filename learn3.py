def log( func ):
    def wrapper():
        print( 'call %s():'% func.__name__ )
        return func()
    return wrapper
@log
def now():
    print( 'sjdkal' )
now()

