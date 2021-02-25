# Flight-Test-Analysis-Tool
This is a Jupyter Notebook created to graphically analyze Flight Test data. 

The workflow is setup so that you import a csv file into a pandas dataframe and then use bqplot to interact with the data.

Typically, the csv file will contain the data for the whole flight and this tool will enable you to mark and save the data into time slices of interest.

The first interaction is to choose the time slices.
This is done by selecting the start and end times and plotting a zoomed in graph.
If the time slice is set, select save time slice and give it a number and description.
Proceed to define all time slices in the flight.

Once the slices are defined, check for trim shots and stabilization by looking at the curve fitting function to the data.

Data can also be filtered before exported. There is a low-pass Butterworth filter and peak-shaver filters that enable spurious data points to be removed.

Time slices can be saved to disk, containing all parameters or just the selected ones.

Finally, a strip chart can be created with the slices/parameters chosen.
