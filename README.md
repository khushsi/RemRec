#Requirement
nltk <br/>
pandas <br/>
sklearn <br/>
openpyxl <br/>

# RemRec
Recommending Remedial Readings Using Student Knowledge States <br/>

# Sample Input:
This file is generated utilizing PFA model - you can utilize any PFA or student model the output required is as below:  <br/>
data/vec/pfa_sample_concepts_knowledge.csv (this is just a sample -- as I am not allowed to share the original data) <br/>
 
Contact me in case you need PFA code  <br/>

##descrition of CSV Fields 

 interaction_id - interaction id - to maintain order of student practice <br/>
 quiz_id - quiz id <br/>
 current_concepts - to check what concepts are associated with this quiz <br/>
 student_id - studentid <br/>
 outcome - 1 for success 0 for failure <br/>
 quiz_section -section the quiz belongs to  <br/>
 concepts - other columns are concepts with each column with probability of student failure on that KC <br/>

#
recommendation.py - to find the matching <br/>
evaluate.py to generate output <br/>

#
If you are interested in dataset for this paper please send me an email to k.thaker@pitt.edu <br/>
