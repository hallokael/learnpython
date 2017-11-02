import asyncio
@asyncio.coroutine
def hello():
    print('hello')
    r=yield from asyncio.sleep(1)
    print("hello again")
#loop=asyncio.get_event_loop()
#loop.run_until_complete(hello())
#loop.close()
@asyncio.coroutine
def wget(host):
    print('wget %s'%host)
    connect=asyncio.open_connection(host,80)
    print('where-1' + host)
    reader,writer=yield from connect
    print('where0' + host)
    header='GET / HTTP/1.0\r\nHost:%s\r\n\r\n'%host
    writer.write(header.encode('utf-8'))
    print('where1'+host)
    yield from writer.drain()
    print('where2'+host)
    while True:
        line=yield from reader.readline()
        if line==b'\r\n':
            break
        print('%s header > %s'%(host,line.decode('utf-8').rstrip()))
    writer.close()
async def wget2(host):
    print('wget %s'%host)
    connect=asyncio.open_connection(host,80)
    print('where-1' + host)
    reader,writer=await connect
    print('where0' + host)
    header='GET / HTTP/1.0\r\nHost:%s\r\n\r\n'%host
    writer.write(header.encode('utf-8'))
    print('where1'+host)
    await writer.drain()
    print('where2'+host)
    while True:
        line=await reader.readline()
        if line==b'\r\n':
            break
        print('%s header > %s'%(host,line.decode('utf-8').rstrip()))
    writer.close()
loop=asyncio.get_event_loop()
tasks=[wget2(host) for host in ['www.sina.com.cn','www.sohu.com','www.163.com']]
loop.run_until_complete(asyncio.wait(tasks))
loop.close()