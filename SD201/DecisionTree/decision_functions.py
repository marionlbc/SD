import os
import csv
import numpy as np
from copy import deepcopy

### Question 1.a

def BuildDecisionTree(file, minNum):
	"""Each node is represented by its feature or class, its level, its data, its gini and its children.
	node = [feature / class, level, data, gini, child1, child2] """
	with open(file, newline='') as f:
		data_D = list(csv.reader(f))
	attributes_names=data_D[0]
	data_D.pop(0)
	data_D = np.array(data_D).astype(int)
	"""First, we create the set of attributes as a list : [attribute1, valuemin_1, valuemax_1, ..., attributeN, valuemin_1, valuemax_N]"""
	attributes_A=[]
	index=0
	for a in attributes_names:
		min=10
		max=0
		attributes_A.append(a)
		for i in range(1,len(data_D)):
			if data_D[i][index]>max: 		#search the max value of each attribute
				max = data_D[i][index]
			elif data_D[i][index]<min:
				min = data_D[i][index]
		attributes_A.append(min)
		attributes_A.append(max)
		index+=1
	"""Then we can create the tree, and it starts with the root..."""
	root = ['',0]
	return Build(data_D, attributes_A, minNum, root)


def Build(data, attributes, minNum, node):
	level = node[1]
	compteur_0 = 0
	"""To begin with, we count the number of record in the class 0 and in the class 1 (upon the attribute "survived")"""
	for datum in data :
		if datum[3]==0:
			compteur_0+=1
	compteur_1=len(data)-compteur_0
	"""We check if the current node is a leaf : to this end, we check if all the datum of the data are in the same class or if the size of the data is below minNum"""
	if (compteur_0==0) or (compteur_0==len(data)) or (len(data)<minNum): #conditions to stop the recursive function
		if compteur_0 >= compteur_1:
			return [0, level, data, 1-(compteur_0/len(data))**2-(compteur_1/len(data))**2, 'leaf']
		else :
			return [1, level, data, 1-(compteur_0/len(data))**2-(compteur_1/len(data))**2, 'leaf']
	"""At this state of the algorithm, we know that we have to split : let's calculate the gini of every possible split !"""
	n = len(attributes) - 3
	all_potentialchildren_left, all_potentialchildren_right, all_listginis = [],[],[] #for each list, list[i] refers to the information of the i-th attribute that hasn't been used yet
	min_gini=1
	for i in range(0,n,3): 					#filling the three lists above in order to examine all the different splits possibles
		a, min, max = attributes[i], attributes[i+1], attributes[i+2]
		potentialchildren_left, potentialchildren_right, listginis=[],[],[]
		for j in range(min+1,max+1):
			data_left, data_right = [],[]
			compteur_0_left, compteur_0_right = 0,0
			for datum in data:
				if datum[i//3] < j: #we sort the data under the atttribute
					data_left.append(datum)
					if datum[3]==0:
						compteur_0_left+=1
				else :
					data_right.append(datum)
					if datum[3]==0:
						compteur_0_right+=1
			if (len(data_right)>0) and (len(data_left)>0): #if one of the list is empty it means that the data is already sorted upon the current attribute
				compteur_1_left = len(data_left) - compteur_0_left
				compteur_1_right = len(data_right) - compteur_0_right
				gini_left = 1 - (compteur_0_left/len(data_left))**2 - (compteur_1_left/len(data_left))**2
				gini_right = 1 - (compteur_0_right/len(data_right))**2 - (compteur_1_right/len(data_right))**2
				gini_split = (len(data_left)/len(data))*gini_left + (len(data_right)/len(data))*gini_right
			else :
				data_left.append(data[0])
				data_right.append(data[0])
				gini_split=1
			listginis.append(gini_split)
			potentialchildren_left.append(data_left)
			potentialchildren_right.append(data_right)
			if gini_split<min_gini: #we keep the information of the best gini in order to find it after we have calculated all the possible ginis
				min_gini = gini_split
				index_best_attribute = i//3 #index of the attribute with the smallest gini
				index_best_children = len(listginis)-1 #index of the smallest gini of the children
				constraint = [k for k in range(min,j)]
		all_potentialchildren_left.append(potentialchildren_left) #each element is a list of data of a potential left child for the node
		all_potentialchildren_right.append(potentialchildren_right)
		all_listginis.append(listginis)
	if min_gini==1: 							#if all the attributes have been already used, we stop the function
		if compteur_0 >= compteur_1:
			return [0, level, data, 1-(compteur_0/len(data))**2-(compteur_1/len(data))**2, 'leaf']
		else :
			return [1, level, data, 1-(compteur_0/len(data))**2-(compteur_1/len(data))**2, 'leaf']
	"""Now that we now the attribute and its condition : we split !"""
	node = [(attributes[index_best_attribute*3],constraint),level, data, 1-(compteur_0/len(data))**2-(compteur_1/len(data))**2]
	level+=1
	node.append(Build(all_potentialchildren_left[index_best_attribute][index_best_children],attributes, minNum, ['', level]))
	node.append(Build(all_potentialchildren_right[index_best_attribute][index_best_children],attributes, minNum, ['', level]))
	return node


### Question 1.b

def printDecisionTree(tree, minNum):
	print("Root")
	print("Level", 0)
	print("Feature", tree[0][0], tree[0][1])
	print("Gini", tree[3])
	print("")
	next_nodes = [tree[4],tree[5]]
	while next_nodes!=[]:
		new_next_nodes = []
		for node in next_nodes :
			if node[4]=='leaf':
				print('Leaf')
				print("Level", node[1])
				print("Class", node[0])
				print("Gini", node[3])
			else :
				print("Intermediate")
				print("Level", node[1])
				print("Feature", node[0][0], node[0][1])
				print("Gini", node[3])
				new_next_nodes.append(node[4])
				new_next_nodes.append(node[5])
			if node!= next_nodes[len(next_nodes)-1]:
				print('*****')
		next_nodes=new_next_nodes
		print("")


### Question 2

def generalizationError(data, tree, alpha):
	nb_leaves=0
	"""First, we count the number of leaves in order to have the complexity of our tree"""
	next_nodes = [tree[4],tree[5]]
	while next_nodes!=[]:
		new_next_nodes = []
		for node in next_nodes :
			if node[4]=='leaf':
				nb_leaves+=1
			else :
				new_next_nodes.append(node[4])
				new_next_nodes.append(node[5])
		next_nodes=new_next_nodes
	"""Then, we count the number of errors"""
	nb_errors = 0
	for datum in data:
		if testingData(datum, tree)==False:
			nb_errors+=1
	return (nb_errors+alpha*nb_leaves)/len(data)

def testingData(datum, tree):
	"""We go down the tree upon the values of the datum and when we reach a leaf, we check if the class of the datum is the class of the leaf"""
	node = tree
	level = 0
	datum = np.array(datum).astype(int)
	while node[4]!='leaf':
		attribute, condition = node[0][0], max(node[0][1])+1
		if attribute=='Sex': #choose the right index of the datum
			index=0
		elif attribute=='Pclass':
			index=1
		elif attribute=='Embarked':
			index=2
		if datum[index]<condition: #check if it belong the the left child ou right child
			node = node[4]
		else :
			node = node[5]
		level+=1
	if node[0]==datum[3] : #check if the class of the datum is the class of the leaf
		return True
	else :
		return False


###Question 3

def pruneTree(tree, alpha, minNum):
	lvl=0
	next_nodes =[[tree]]
	pasfini=True
	while (pasfini):
		all_children = []
		for node in next_nodes[lvl]:
			if type(node)==list and node[4]!='leaf':
				all_children.append(node[4])
				all_children.append(node[5])
				pasfini=True
			else :
				all_children.append('leaf')
				pasfini=False
		next_nodes.append(all_children)
		lvl+=1
	for current_lvl in range(lvl-1, 0, -1):
		for node_to_leaf in next_nodes[current_lvl]:
			"""For each node : first, we transform it into a leaf"""
			path=[]
			if node_to_leaf!='leaf' and node_to_leaf[4]!='leaf':
				compteur_0 = 0
				data = node_to_leaf[2]
				for datum in data :
					if datum[3]==0:
						compteur_0+=1
				compteur_1=len(data)-compteur_0
				if compteur_0 >= compteur_1:
					new_leaf = [0, node_to_leaf[1], data, 1-(compteur_0/len(data))**2-(compteur_1/len(data))**2, 'leaf']
				else :
					new_leaf = [1, node_to_leaf[1], data, 1-(compteur_0/len(data))**2-(compteur_1/len(data))**2, 'leaf']
				"""Then we find its path in the tree"""
				level=current_lvl
				indice = next_nodes[level].index(node_to_leaf)
				while level!=0 : #stop when we reach level 0 (= the root)
					if indice%2==0: #check if it belongs to the left branch of its parent
						path.append(4)
					else :
						path.append(5)
					level-=1
					indice = indice//2
					for node in next_nodes[level]:
						if node=='leaf':
							indice+=1
						elif next_nodes[level].index(node)==indice:
							break
				path=path[::-1]
				"""Once we have the path, we reach the node to change and put a leaf.
				Finally we can compare the tree VS the pruned tree."""
				pruned_tree = changeNode(new_leaf, path, tree)
				if generalizationError(tree[2], pruned_tree, alpha) < generalizationError(tree[2], tree, alpha):
					tree = pruned_tree
	printDecisionTree(tree, minNum)

def changeNode(node, path, old_list):
	"""Here, I couldn't find another way to reach the element in my tree but to do a case disjonction. I tried to make my list "tree" into an array and use tree[tuple(path)] but it didn't work because my list hasn't the correct shape"""
	prunedList=deepcopy(old_list)
	if len(path)==1:
		prunedList[path[0]]= node
		return prunedList
	elif len(path)==2:
		prunedList[path[0]][path[1]]= node
		return prunedList
	elif len(path)==3:
		prunedList[path[0]][path[1]][path[2]]= node
		return prunedList
	elif len(path)==4:
		prunedList[path[0]][path[1]][path[2]][path[3]]= node
		return prunedList
	elif len(path)==5:
		prunedList[path[0]][path[1]][path[2]][path[3]][path[4]]=node
		return prunedList
	elif len(path)==6:
		prunedList[path[0]][path[1]][path[2]][path[3]][path[4]][path[5]]=node
		return prunedList
















