class Student( object ):
    def __init__(self,name,score):
        self.__name=name
        self.__score=score
    def print_score(self ):
        print('%s: %s'%( self.__name,self.__score ))
bart=Student( 'Bart Simp',59 )
lisa=Student( 'Lisa Simp',87 )
bart.print_score()
print( bart._Student__name )
bart.__name
