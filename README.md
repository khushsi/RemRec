#Requirement
nltk
pandas
sklearn
openpyxl

# RemRec
Recommending Remedial Readings Using Student Knowledge States

# Sample Input:
This file is generated utilizing PFA model - you can utilize any PFA or student model the output required is as below:
data/vec/pfa_sample_concepts_knowledge.csv
 
Contact me in case you need PFA code

##descrition of CSV Fields

 interaction_id - interaction id - to maintain order of student practice
 quiz_id - quiz id
 current_concepts - to check what concepts are associated with this quiz
 student_id - studentid
 outcome - 1 for success 0 for failure
 quiz_section -section the quiz belongs to 
 concepts - other columns are concepts with each column with probability of student failure on that KC

#
recommendation.py - to find the matching
evaluate.py to generate output
