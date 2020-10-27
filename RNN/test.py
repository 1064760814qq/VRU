# class Person(object):
#     def __init__(self, name, gender, age):
#         self.name = name
#         self.gender = gender
#         self.age = age
#
#
# class Student(Person):
#     def __init__(self, name, gender, age, school, score):
#         #super(Student,self).__init__(name,gender,age)
#         self.name = name.upper()
#         self.gender = gender.upper()
#         self.school = school
#         self.score = score
#
#
# s = Student('Alice', 'female', 18, 'Middle school', 87)
# print(s.school)
# print(s.name)
import torch

import torch.nn as nn
import torch
x = torch.rand(10,24,100)
lstm = nn.LSTM(100,16,bidirectional=True)
output,(h,c) = lstm(x)
print(output)
print(h.size())
print(c.size())

# b=torch.arange(1,19)
# b=b.view(2,9,-1)
# print(b)