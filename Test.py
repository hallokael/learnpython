def FillZero(num):
    if len(num)<8:
        #print(len(num))
        for x in range(8-len(num)):
            num='0'+num
    return num
def Tri(num):
    n,a=0,[1]
    for i in range(0,num):
        yield a
        a=[x+y for x,y in zip(a+[0],[0]+a)]
    return "finish"
def char2num(s):
    print(s)
    return {'0':0,'1':1,'2':2}[s]
def not_empty(s):
    return s and s.strip()
def odd_it():
    n=1
    while True:
        n=n+2
        yield n
def not_di( n ):
    return lambda x:x%n>0
def Fi():
    yield 2
    it=odd_it()
    while True:
        n=next(it)
        yield n
        it=filter( not_di(n),it )
def is_palindrome( n ):
    t=str( n )
    return n and str( n )==str( n )[::-1]
    #print( t,t[:int( len( t )/2 ) ],t[-int( len( t )/2 ):] )
    #return t[:int( len( t )/2 ) ] == t[-int( len( t )/2 ):]
#output = filter(is_palindrome, range(1, 1000))
#print(list(output))



