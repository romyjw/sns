import re

def matlab_to_numpy(equations_string):
	equations_string = re.sub(r'\^', '**', equations_string, count=0, flags=0)
	equations_string = re.sub(r'sin', 'np.sin', equations_string, count=0, flags=0)
	equations_string = re.sub(r'cos', 'np.cos', equations_string, count=0, flags=0)
	equations_string = re.sub(r'\n', ' ', equations_string, count=0, flags=0)
	equations_string = re.sub(r'pi', 'np.pi', equations_string, count=0, flags=0)
	
	equations_string = re.sub(r'^\s+', '', equations_string, flags=re.MULTILINE)
	
	return equations_string.split(';')

def sub_variable(expression_string, original_var='a', new_var='b'):
	expression_string = re.sub(original_var, new_var, expression_string, count=0, flags=0)
	return expression_string

