try:
    print( 'try...' )
    r=10/int( 'a' )
    print('result:',r)
except ZeroDivisionError as e:
    print('except:',e)
except ValueError as e:
    print('ValueError:',e)
else:
    print('no error')
finally:
    print('finally...')
print('END')

import logging

def foo(s):
    return 10 / int(s)

def bar(s):
    return foo(s) * 2

def main():
    try:
        bar('0')
    except Exception as e:
        logging.exception(e)

main()
print( 'END' )