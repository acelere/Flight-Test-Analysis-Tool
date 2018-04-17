# Flight-Test-Analysis-Tool
This is a Jupyter Notebook created to analyze Flight Test data. It has the most usual graphing tools.

The workflow is setup so that you import a csv file into a pandas dataframe and then use bqplot to interact with the data.

The csv file should contain all data, from the whole flight.

The first interaction is to choose the time slices.
This is done by selecting the start and end times and plotting a zoomed in graph.
If the time slice is set, select save time slice and give it a number and description.
Proceed to define all time slices in the flight.

Once the slices are defined, the next set of cells help you check for trim shots and stabilization by offering a curve fitting function to the data.

Time slices can be saved to disk.

Finally, a strip chart can be created with the slices/parameters chosen.
