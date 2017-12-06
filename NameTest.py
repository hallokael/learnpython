class Person:
    name = []
p1 = Person()
p2 = Person()
p1.name.append(1)
p2.name.append(2)
print(p1.name)
print(p2.name)
print(Person.name)
