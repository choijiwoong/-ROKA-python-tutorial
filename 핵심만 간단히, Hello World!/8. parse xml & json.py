#[1. make XML]
#Make node
from xml.etree.ElementTree import Element, dump, ElementTree

node1=Element("first")
node1.text="안녕"
dump(node1)
print()

#Append node to node
root=Element("xml")
node1=Element("first")#or SubElement(root, "first").text="Hello"
node1.text="Hello"
root.append(node1)

node2=Element("second")
node2.text="Hi"
root.append(node2)#or SubElement(root, "second").text="Hi"

dump(root)
print()

#Append attribute
root=Element("xml", kind="language")#or root=Element("xml"); root[kind]="language";
node1=Element("First")
node1.text="Happy? or not a problem.."
root.append(node1)

node2=Element("Second")
node2.text="Sad? or a problem.."
root.append(node2)

dump(root)
print()

#Make it look like good by \n, space
from xml.etree.ElementTree import Element, dump

root=Element("xml", kind="language")

node1=Element("first_element")
node1.text="mountain"
root.append(node1)

node2=Element("second_element")
node2.text="water"
root.append(node2)

def indent(elem, level=0):#Add proper '\n', ' ' for each elements
    i='\n'+level*"  "#print as much as element's level
    if len(elem):#is valid
        if not elem.text or elem.text.strip():#empty elem
            elem.text=i+"  "
        if not elem.tail or not elem.tail.strip():#empty tail(tail is char after it's text)
            elem.tail=i
            
        for elem in elem:
            indent(elem, level+1)#recursive
        if not elem.tail or not elem.tail.strip():#all empty
            elem.tail=i#default
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail=i#default
indent(root)
dump(root)

#Write as xml file
ElementTree(root).write("note.xml")


#[2. parse XML]
""" data.xml
<?xml version="1.0"?>
<data>
<student>
    <name>peter</name>
    <age>24</age>
    <score math="80" english="97"/>
</student>
<student>
    <name>elgar</name>
    <age>21</age>
    <score math="67" english="56"/>
</student>
<student>
    <name>hong</name>
    <age>36</age>
    <score math="76" english="81"/>// score의 값은 속성값으로 저장되어 있다.
</student>
</data>
"""

from xml.etree.ElementTree import parse
tree=parse('test.xml')#get xml file
root=tree.getroot()#get rootnode

student=root.findall("student")#get matched all node

name=[x.findtext("name") for x in student]
age=[x.findtext("age") for x in student]
score=[x.find("score").attrib for x in student]#.attrib

print(name, age, score)

#for get matched first Node
#student.find("name")#<name>peter</name>. if we want to value, use .text like student.find("name").text
#for get matched first node's value
#student.findtext("name")
#for get matched first node's attrubute
#student.findtext("score").attrib


#[3. json parsing]
import json
from collections import OrderedDict
from pprint import pprint#for visibility

with open('test.json', 'r') as f:
    data=json.load(f, object_pairs_hook=OrderedDict)#load as OrderedDict
pprint(data)
