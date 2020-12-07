#!/usr/bin/env python
# coding: utf-8

# In[35]:


try:
	from decision_functions import BuildDecisionTree
	tree=BuildDecisionTree("data.csv",5)
except Exception as e:
	raise e

try:
	from decision_functions import printDecisionTree
	import sys
	save=sys.stdout
	output = open('output_tree.txt', 'w')
	sys.stdout=output
	printDecisionTree(tree,5)
	sys.stdout=save
	output.close()
except Exception as e:
	raise e

try:
	from decision_functions import generalizationError
	genError = open('generalization_error.txt', 'w')
	genError.write("The generalization error is : ")
	genError.write(str(generalizationError(tree[2],tree,0.5)))
	genError.close()
except Exception as e:
	raise e

try:
	from decision_functions import pruneTree
	pruneTree(tree, 0.5, 5)
	save=sys.stdout
	postpruned = open('postpruned_tree.txt', 'w')
	sys.stdout=postpruned
	pruneTree(tree,0.5,5)
	sys.stdout=save
	postpruned.close()
except Exception as e:
	raise e


# In[ ]:




