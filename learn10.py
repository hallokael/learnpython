import pickle
d=dict( name='Bob',age=20,score=88 )
#print(pickle.dumps(d))
import json
print( json.dumps(d) )
e=json.loads(json.dumps(d))
print( type( e ) )
class Student(object):
    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score
s=Student( 'Bob',20,88 )
print( json.dumps(s,default=lambda obj:obj.__dict__) )
jstr=json.dumps(s,default=lambda obj:obj.__dict__)
print( type( jstr ) )
def d2s( strd ):
    return Student( strd['name'],strd['age'],strd['score'] )
print(type( json.loads(jstr)),type( d2s( json.loads(jstr) ) ) )
print( type( json.loads( jstr,object_hook=d2s ) ) )