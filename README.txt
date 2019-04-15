##Author: Neil Acharya 
##CS 4641 Project 4 Submission

Link to code: https://github.com/nacharya114/mlproject4
###########################################################
# Instruction
###########################################################

This project code is run using BURLAP java-maven library. 

The source code can be found in src.main.java.rlDriver, where SmallGridWorldAnalysis are the tests for the first MDP,
and GridMazeAnalysis are for my larger MDP.

Additionally, maze_maker.py was used to generate the grid world used in GridMazeAnalysis. 

To run the code either use an IDE with maven utilities installed, or use the command line input:
`mvn compile exec:java -Dexec.mainClass="rlDriver.<ClassName>" `