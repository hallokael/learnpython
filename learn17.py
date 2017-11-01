def consumer():
    r=''
    while True:
        n=yield r
        if not n:
            return
        print('consume %s'%n)
def produce(c):
    c.send(None)
    n=0
    while n<5:
        n=n+1
        print('producing %s'%n)
        r=c.send(n)
    c.close()
c=consumer()
produce(c)
