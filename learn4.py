class Student( object ):
    def __init__(self,name,score):
        self.__name=name
        self.__score=score
    def print_score(self ):
        print('%s: %s'%( self.__name,self.__score ))
bart=Student( 'Bart Simp',59 )
lisa=Student( 'Lisa Simp',87 )
#bart.print_score()
#print( bart._Student__name )
#bart.__name
class Animal( object ):
    def run(self):
        print('Animal is running')
class Dog( Animal ):
    def run(self  ):
        print('Dog is running !!')
dog=Dog()
#dog.run()
def run_twice( animal ):
    animal.run()
    animal.run()
run_twice(dog)

