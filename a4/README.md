# a4

This assignment will allow you to do a more open-ended exploration of online social networking. The goal is to let you use some of the tools we've learned in class on your own. There are some requirements to constrain your work, defined below.

To grade your project, I will run the following commands:
```
python collect.py
python cluster.py
python classify.py
python summarize.py
```
So, your a4 folder should have (at least) those four files. **Please check that your files are named correctly, including using lower case letters!!!**

Here is what each script should do:

- `collect.py`: This should collect data used in your analysis. This may mean submitting queries to Twitter or Facebook API, or scraping webpages. The data should be raw and come directly from the original source -- that is, you may not use data that others have already collected and processed for you (e.g., you may not use [SNAP](http://snap.stanford.edu/data/index.html) datasets). Running this script should create a file or files containing the data that you need for the subsequent phases of analysis.
- `cluster.py`: This should read the data collected in the previous steps and use any community detection algorithm to cluster users into communities. You may write any files you need to save the results.
- `classify.py`: This should classify your data along any dimension of your choosing (e.g., sentiment, gender, spam, etc.). You may write any files you need to save the results.
- `summarize.py`: This should read the output of the previous methods to write a textfile called `summary.txt` containing the following entries:
  - Number of users collected:
  - Number of messages collected:
  - Number of communities discovered:
  - Average number of users per community:
  - Number of instances per class found:
  - One example from each class:

Additionally, you should create a plain text file called 'description.txt' that contains a brief summary of what your code does and any conclusions you have made from the analysis (3-5 paragraphs).

Other notes:

- You may use any of the algorithms in scikit-learn, networkx, scipy, numpy, nltk to perform your analysis. You do not need to implement the methods from scratch.
- It is expected that when I run your `collect.py` script, I may get different data than you collected when you tested your code. While the final results of the analysis may differ, your scripts should still work on new datasets.
- You may checkin to Github any configuration or data files that your code needs. For example, if you've used manually annotated training data to fit a classifier, you may store that in Github. However, you should not store large data files (e.g., >50Mb). However, please ensure that your code will run using the commands above. Ensure that you use *relative*, not *absolute* paths when needed. (E.g., don't put "C:/Aron/data" as a path.) I recommend checking that your code works on another system prior to submission.
