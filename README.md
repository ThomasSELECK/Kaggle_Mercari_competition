# Kaggle_Mercari_competition

This repository contains the solution developed for the Kaggle Mercari competition that finished 46th on Private Leaderboard.

# Solution hardware requirements

This solution was designed to run in a hardware constrained environment (Kaggle's kernels) with the following constraints:
 * 4 CPU cores
 * 16 GB RAM
 * 1 GB disk space
 
The solution takes 1 hour to run on the Kaggle environment for the first test set (Public Leaderboard) and takes 1 hour and 33 minutes for the second test set (Private Leaderboard).

# How to run the code
1. Make sure you have Python 3.6 installed and all required packages installed. You can install the packages using the following command:
	```
	pip install -r requirements.txt
	```

2. Make sure the paths indicated in 'files_paths.py' corresponds to the ones of your system (where the data are stored, where the submission file will be output).
	
3. Run the file 'LeaderboardScript.py' in a terminal
	```
	python LeaderboardScript.py
	```
	
4. That's it!