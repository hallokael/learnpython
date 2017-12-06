with open('tee','r') as f:
    print(f.readline())
with open('tee','w') as f:
    f.write('hello,world!')
from io import StringIO
f=StringIO( 'hello\nhi\nbye' )
while True:
    s=f.readline()
    if s== '':
        break
    print( s.strip() )
import os
print( os.name,os.environ )
print( [x for x in os.listdir('.') if os.path.isdir(x)] )
print( [x for x in os.listdir('.')] )
print( [x for x in os.listdir('.') if x[-3:]=='.py' ] )
print(os.listdir('.')==[x for x in os.listdir('.')])
